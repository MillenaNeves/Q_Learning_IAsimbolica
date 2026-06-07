# Projeto Q-Learning

## Sobre o Projeto

Este repositório contém a implementação do algoritmo **Q-Learning**, uma técnica de Aprendizado por Reforço, desenvolvida para controlar o personagem **Amongois** em um ambiente de plataformas.

O objetivo do agente é aprender, por meio de interação com o ambiente, a melhor sequência de ações para alcançar a plataforma final do mapa, maximizando as recompensas recebidas e minimizando penalidades ao longo do percurso.

---

## Objetivos

* Implementar o algoritmo Q-Learning no arquivo `client.py`;
* Estabelecer comunicação com o jogo através de sockets TCP;
* Treinar um agente capaz de navegar autonomamente pelo ambiente;
* Gerar uma Q-Table contendo os valores aprendidos para todos os estados possíveis;
* Documentar e organizar o código para facilitar manutenção e avaliação.

---

## Descrição do Ambiente

O jogo consiste em um conjunto de plataformas pelas quais o personagem deve se deslocar até alcançar o bloco objetivo.

O ambiente apresenta características estocásticas, ou seja, algumas ações possuem comportamento não determinístico. Em determinadas situações, um comando de avanço pode resultar em um deslocamento inesperado para outra direção, tornando o problema mais desafiador e adequado para técnicas de Aprendizado por Reforço.

### Ações disponíveis

| Ação    | Descrição             |
| ------- | --------------------- |
| `left`  | Girar para a esquerda |
| `right` | Girar para a direita  |
| `jump`  | Pular para frente     |

---

## Representação dos estados

O ambiente possui **96 estados possíveis**, resultantes da combinação de:

* 24 plataformas;
* 4 direções possíveis.

Cada estado é representado por um vetor binário de 7 bits:

* 5 bits → plataforma atual (0 a 23);
* 2 bits → direção atual.

### Direções

| Bits | Direção |
| ---- | ------- |
| 00   | Norte   |
| 01   | Leste   |
| 10   | Sul     |
| 11   | Oeste   |

---

## Sistema de recompensas

As recompensas são fornecidas pelo ambiente após cada ação executada.

* Penalidades entre **-14 e -1** para estados intermediários;
* Recompensa de **+100** ao atingir a plataforma objetivo.

Esse sistema incentiva o agente a encontrar caminhos eficientes até o destino final.

---

## Estrutura do projeto

```text
.
├── client.py
├── connection.py
├── resultado.txt
├── README.md
```

### Arquivos

* **client.py**: implementação do algoritmo Q-Learning;
* **connection.py**: interface de comunicação com o jogo;
* **resultado.txt**: Q-Table gerada após o treinamento;
* **README.md**: documentação do projeto.

---

## 🚀 Como Executar

### 1. Inicie o jogo

Execute o arquivo do jogo disponibilizado pelo professor.

### 2. Configure a conexão

Verifique se a porta utilizada no código corresponde à porta configurada no jogo (por padrão, 2037).

### 3. Execute o agente

```bash
python client.py
```

### 4. Treinamento

Para treinar: defina TRAIN_MODE = True em client.py
Para testar a política aprendida: defina TRAIN_MODE = False

---

## Algoritmo Utilizado

O agente utiliza o algoritmo **Q-Learning**, que atualiza iterativamente os valores da Q-Table utilizando a equação:

Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

onde:

* `s` = estado atual;
* `a` = ação executada;
* `r` = recompensa recebida;
* `s'` = próximo estado;
* `α` = taxa de aprendizado;
* `γ` = fator de desconto.

O treinamento busca estimar os melhores valores para cada par estado-ação, permitindo que o agente tome decisões cada vez mais eficientes.

---

## 👩‍💻 Equipe

* Anysabele de Paula Barbosa Santos
* Maria Clara Pereira Gonçalves
* Millena Ferreira Marçal das Neves

---

## 🎓 Disciplina

Projeto acadêmico desenvolvido para a disciplina de Inteligência Artificial Simbólica, com foco em aprendizado por reforço utilizando o algoritmo Q-Learning.
 
