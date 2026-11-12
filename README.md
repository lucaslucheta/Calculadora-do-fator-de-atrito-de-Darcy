# CALCULADORA WEB - FATOR DE ATRITO DE DARCY / DIAGRAMA DE MOODY
Aplicação web interativa para cálculo do fator de atrito de Darcy pelo Diagrama de Moody. Desenvolvida em Python, utiliza Streamlit, NumPy e SciPy para calcular, classificar regimes de escoamento, converter unidades de rugosidade e visualizar graficamente o comportamento hidráulico em tubulações.

---

## REQUISITOS
- Python 3.11 ou superior
- Conexao com a internet apenas para instalar as bibliotecas na primeira vez

---

## INSTALACAO NO WINDOWS
1. Extraia a pasta/arquivo ZIP para um local de sua preferencia.
2. Abra o Prompt de Comando (CMD) dentro da pasta extraida.
   Dica: no Explorador de Arquivos, entre na pasta, clique na barra de endereco,
   digite cmd e pressione Enter.
3. Execute:
   py -m pip install -r requirements.txt

---

##EXECUCAO
4. Execute:
   py -m streamlit run app.py
5. O Streamlit abrira a aplicacao no navegador. Se nao abrir automaticamente,
   copie o endereco exibido no CMD (normalmente http://localhost:8501) e cole
   no navegador.
6. Para encerrar, volte ao CMD e pressione Ctrl+C.
7. Também é possível executar pelo link: https://calculadora-do-fator-de-atrito-de-darcy.streamlit.app/

---

## FUNCIONALIDADES
- Entrada de Reynolds.
- Rugosidade relativa direta ou rugosidade absoluta + diametro.
- Unidades independentes para rugosidade e diametro: mm, cm, m e pol.
- Calculo nos regimes laminar, transicional e turbulento.
- Solucao numerica de Prandtl-Karman e Colebrook-White com Brent.
- Criterio configuravel para regiao totalmente rugosa.
- Grafico do Diagrama de Moody com destaque do ponto calculado.

---

## NOTA TECNICA
A faixa 2300 <= Re <= 4000 e instavel. O app usa uma interpolacao suave como
estimativa, nao como correlacao universal. O criterio de plena turbulencia
considera o termo de Reynolds <= 1% do termo de rugosidade na equacao de
Colebrook-White.

---

## Notas de Desenvolvimento e Apoio Tecnológico

Este projeto foi concebido, estruturado e validado por **Lucas Reis Lucheta**. Durante o processo de desenvolvimento e programação da interface em Streamlit, foram utilizadas ferramentas de Inteligência Artificial (como o **Gemini**) como suporte técnico de mentoria, estruturação de código e boas práticas de desenvolvimento.

---

## 👨‍💻 Autor

Desenvolvido por **Lucas Reis Lucheta**  
Estudante de Engenharia Química | EEL-USP  
[LinkedIn](https://www.linkedin.com/in/lucas-reis-lucheta) | [GitHub](https://github.com/lucaslucheta)

