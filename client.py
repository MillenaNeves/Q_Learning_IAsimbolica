import os
import random
import numpy as np
import connection as cn

# Constantes do ambiente

ACTIONS        = ["left", "right", "jump"]   # ordem obrigatória da Q-table
ACTION_TO_IDX  = {a: i for i, a in enumerate(ACTIONS)}

NUM_PLATFORMS  = 24
NUM_DIRECTIONS = 4
NUM_STATES     = NUM_PLATFORMS * NUM_DIRECTIONS   # 96
NUM_ACTIONS    = len(ACTIONS)                     # 3

# Hiperparâmetros

ALPHA          = 0.7     # taxa de aprendizado 
GAMMA          = 0.99    # desconto: alto porque a recompensa final é distante
# EPSILON_START: quando há Q-table salva, começa já com exploração baixa
# p refinar oq foi aprendido, não explorar do zero
EPSILON_START  = 0.15    
EPSILON_END    = 0.05    # mínimo de exploração
EPSILON_DECAY  = 0.998   # decaimento lento p manter alguma exploração

EPISODES       = 3000    # episódios de refinamento
MAX_STEPS      = 500     # passos por episódio: mais passos dá mais tempo para escapar de loops
SAVE_EVERY     = 100     # salvar Q-table a cada N episódios
EP_TESTS       = 20      # episódios de avaliação da política aprendida

SUCCESS_REWARD = 100     # recompensa do bloco-objetivo 
FALL_THRESHOLD = 5       # queda detectada se plataforma recuar mais que isso

# Arquivo de saída

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
QTABLE_FILE = os.path.join(BASE_DIR, "resultado.txt")

# Função utilitária: decodificação do estado

def decode_state(raw) -> int:
    # o servidor envia o estado como string '0b0000100' (formato bin() do python)
    # removi o prefixo '0b' p extrair só os dígitos binários
    s = str(raw).strip()
    if s.startswith('0b') or s.startswith('0B'):
        s = s[2:]                          # remove prefixo '0b'

    bits = ''.join(ch for ch in s if ch in '01')  # só '0' e '1'
    bits = bits.zfill(7)[-7:]             # garante exatamente 7 bits

    if not bits or not all(ch in '01' for ch in bits):
        print(f"[AVISO] Estado inesperado: '{raw}' -> fallback para estado 0")
        return 0

    platform  = int(bits[:5], 2)   # 5 bits mais significativos
    direction = int(bits[5:],  2)  # 2 bits menos significativos

    if not (0 <= platform < NUM_PLATFORMS and 0 <= direction < NUM_DIRECTIONS):
        print(f"[AVISO] Fora do range: plat={platform} dir={direction} → fallback 0")
        return 0

    return platform * NUM_DIRECTIONS + direction

# Agente Q-Learning

class QLearningAgent:
    """Implementação do algoritmo Q-Learning tabular para o Amongois."""

    def __init__(self):
        self.alpha   = ALPHA
        self.gamma   = GAMMA
        self.epsilon = EPSILON_START
        self.q_table = self._load_q_table()

    # Persistência da Q-table

    def _load_q_table(self) -> np.ndarray:
        """Carrega a Q-table do disco; cria uma zerada se não existir."""
        if os.path.exists(QTABLE_FILE):
            try:
                data = np.loadtxt(QTABLE_FILE)
                if data.shape == (NUM_STATES, NUM_ACTIONS):
                    print(f"[INFO] Q-table carregada de: {QTABLE_FILE}")
                    return data
                else:
                    print(f"[AVISO] Shape incorreto {data.shape}. Criando Q-table zerada.")
            except Exception as e:
                print(f"[AVISO] Falha ao carregar Q-table: {e}. Criando zerada.")
        print("[INFO] Nenhuma Q-table encontrada. Iniciando do zero.")
        return np.zeros((NUM_STATES, NUM_ACTIONS), dtype=float)

    def save_q_table(self):
        """Salva a Q-table no formato exigido pelo enunciado (sem cabeçalho)."""
        np.savetxt(QTABLE_FILE, self.q_table, fmt="%.6f")
        print(f"[INFO] Q-table salva em: {QTABLE_FILE}")

    # Política ε-greedy
    
    def choose_action(self, state: int) -> str:
        """
        Escolhe uma ação com política ε-greedy.
        - Com probabilidade ε: ação aleatória (exploração).
        - Com probabilidade 1-ε: melhor ação conhecida (explotação).
        """
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return ACTIONS[int(np.argmax(self.q_table[state]))]

    # Atualização da Q-table (regra de Bellman)

    def update(self, state: int, action: str, reward: float,
               next_state: int, done: bool):
        """
        Aplica a equação de Bellman:
            Q(s,a) ← Q(s,a) + α · [r + γ·max_a'Q(s',a') − Q(s,a)]
        Se o episódio terminou (done=True), não há estado futuro.
        """
        idx = ACTION_TO_IDX[action]
        current_q = self.q_table[state, idx]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, idx] += self.alpha * (target - current_q)

    # Treino

    def train(self, sock):
        """
        Loop principal de treino Q-Learning.

        Cada episódio:
          1. Lê o estado REAL do servidor (não sorteia índice aleatório)
          2. Escolhe ação (ε-greedy), envia ao servidor, recebe próximo
             estado e recompensa
          3. Atualiza Q-table
          4. Detecta fim de episódio (vitória ou queda)
          5. Decai ε ao final do episódio
        """
        print("\n=== INICIANDO TREINO ===")
        successes = 0

        for ep in range(1, EPISODES + 1):
            raw_state, _ = cn.get_state_reward(sock, "jump")
            state = decode_state(raw_state)
            done  = False

            visited_train = {}   # p detectar loop no treino tmb

            for step in range(MAX_STEPS):
                action = self.choose_action(state)

                raw_next, raw_reward = cn.get_state_reward(sock, action)
                next_state = decode_state(raw_next)
                reward     = float(raw_reward)

                # Detecção de fim de episódio 
                plat_cur  = state      // NUM_DIRECTIONS
                plat_next = next_state // NUM_DIRECTIONS

                if reward >= SUCCESS_REWARD:
                    done = True
                    successes += 1
                    print(f"  [Ep {ep:4d}] 🎉 OBJETIVO ALCANÇADO (passo {step+1})")

                # Queda: recuo grande de plataforma: aplica penalidade extra
                elif plat_cur > 2 and plat_next < plat_cur - FALL_THRESHOLD:
                    reward = -50   # penalidade de queda p desincentivar ações ruins
                    done = True

                # Loop no treino: forçar fim p não desperdiçar episódio
                visited_train[state] = visited_train.get(state, 0) + 1
                if visited_train[state] >= 15:
                    done = True

                self.update(state, action, reward, next_state, done)
                state = next_state

                if done:
                    break

            # Decai exploração
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

            # Salva periodicamente
            if ep % SAVE_EVERY == 0:
                self.save_q_table()
                pct = successes / ep * 100
                print(f"  [Ep {ep:4d}/{EPISODES}] ε={self.epsilon:.4f} | "
                      f"Sucesso acumulado: {pct:.1f}%")

        self.save_q_table()
        print(f"\n=== TREINO CONCLUÍDO ===")
        print(f"Episódios de sucesso: {successes}/{EPISODES} "
              f"({successes/EPISODES*100:.1f}%)")

    # Teste da política aprendida

    def test_policy(self, sock, ep_tests: int = EP_TESTS):
        """
        Avalia a política aprendida com recuperação de loop.

        O ambiente é estocástico: mesmo a política ótima pode cair
        ocasionalmente. Quando isso ocorre e o agente entra em loop
        em estados ruins (plat=0/1 com Q-values parecidos), usamos
        uma pequena aleatoriedade LOCAL só p escapar, sem alterar
        a Q-table (não é treino)

        EPSILON_TEST = 0.25 significa: 75% greedy, 25% aleatório apenas
        quando estiver em loop (visitou o estado 3+ vezes)
        """
        EPSILON_TEST = 0.25   # aleatoriedade só p escapar de loops
        LOOP_THRESHOLD = 3    # visitas ao mesmo estado antes de randomizar

        print(f"\n=== TESTANDO POLÍTICA ({ep_tests} episódios) ===")
        successes = 0

        for ep in range(1, ep_tests + 1):
            raw_state, _ = cn.get_state_reward(sock, "left")
            state = decode_state(raw_state)
            done  = False
            visited = {}

            for step in range(MAX_STEPS):
                row = self.q_table[state]
                visit_count = visited.get(state, 0)

                # Política de ação:
                # 1) Estado nunca treinado: jump
                # 2) Em loop (visitado 3+ vezes): aleatoriedade para escapar
                # 3) Normal: greedy (melhor Q)
                if np.all(row == row[0]):
                    action = "jump"
                elif visit_count >= LOOP_THRESHOLD and random.random() < EPSILON_TEST:
                    action = random.choice(ACTIONS)
                else:
                    action = ACTIONS[int(np.argmax(row))]

                # Log dos primeiros 20 passos do episódio 1
                if ep == 1 and step < 20:
                    plat = state // NUM_DIRECTIONS
                    dire = state  % NUM_DIRECTIONS
                    dirs = ["N","L","S","O"][dire]
                    print(f"    passo {step+1:3d}: estado={state:2d} "
                          f"(plat={plat} dir={dirs}) ação={action} visits={visit_count} "
                          f"Q={[round(float(v),1) for v in row]}")

                raw_next, raw_reward = cn.get_state_reward(sock, action)
                next_state = decode_state(raw_next)
                reward     = float(raw_reward)

                if reward >= SUCCESS_REWARD:
                    done = True
                    successes += 1
                    print(f"  [Teste {ep:3d}] Sucesso (passo {step+1})")
                    break

                visited[state] = visit_count + 1
                state = next_state

            if not done:
                print(f"  [Teste {ep:3d}] Falhou ({MAX_STEPS} passos)")

        pct = successes / ep_tests * 100
        print(f"\n=== RESULTADO: {successes}/{ep_tests} ({pct:.1f}%) ===")


# Ponto de entrada

if __name__ == "__main__":
    # conecta ao servidor do jogo (porta padrão 2037)
    sock = cn.connect(2037)

    # inicializa o agente (carrega Q-table existente se houver)
    agent = QLearningAgent()

    # Quando for testar mudar TRAIN_MODE:
    #   True: treina o agente e salva a Q-table
    #   False: carrega a Q-table e avalia a política aprendida
    TRAIN_MODE = False

    if TRAIN_MODE:
        agent.train(sock)
    else:
        agent.test_policy(sock, ep_tests=EP_TESTS)