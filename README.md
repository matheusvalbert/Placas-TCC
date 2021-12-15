# Placas TCC

Reconhecedor de placas melhorado do projeto de TCC (Sistema de controle de acesso para condomínios).

Melhoria do treinamento para OpenALPR para reconhecer placas brasileiras em conjunto com um arquivo utilizado para analise e plot de gráficos que auxiliaram na melhoria do treinamento.

## Teste realizado utilizando a ferramenta do OpenALPR
<p align="center">
  <img src="https://github.com/matheusvalbert/Placas-TCC/blob/main/openalpr_img.jpg" />
</p>

Os testes foram realizados utilizando o dataset fornescido pelo OpenALPR: https://github.com/matheusvalbert/train-detector/tree/master/br

# Utilizando apenas a placa mais provavél

Realizar as seguintes substituições:
- O’s com 0’s e 0’s com O’s
- I’s com 1’s e 1’s com I’s
- B’s com 8’s e 8’s com B’s
- S’s com 5’s e 5’s com S’s

Realizando a substituição das letras e números de acordo com sua posição na placa brasileira, conseguimos uma taxa de acerto de 86,84%.

# Outros repositórios do projeto
Servidor: https://github.com/matheusvalbert/Server-TCC

Desktop: https://github.com/matheusvalbert/Desktop-TCC

Mobile: https://github.com/matheusvalbert/Mobile-TCC
