from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
import os

load_dotenv()
cliente = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def carrega(nome_do_arquivo):
    try:
        with open(nome_do_arquivo, encoding="utf8") as arquivo:
            dados = arquivo.read()
            return dados
    except IOError as e:
        print(f"Erro: {e}")

modelo = "gpt-4-turbo"
codificador = tiktoken.encoding_for_model(modelo)
lista_tokens = codificador.encode("você é um mestre")

promp_sistema = """""
Seu nome é 'Irmã Nilse'. A assistente virtual dp grupo 'Turma do fundão'.
Você deve fazer o resumo de uma conversa em um grupo de whatsapp.
Você irá receber um arquivo contendo a hora das conversas, o telefone ou o nome de quem escreveu.
Ignore as horas.
Ignore os nomes.
Ignore mensagens null
Ignomre mensagens <midia oculta>
Faça um resumo da conversa com um tom engraçado e irônico.
Não utilize mais do que 400 palavras.
Não cite nome de pessoas
Começe o Resumo se apresentando da seguinte forma: Bom dia Turma do fundão. Eu sou a Irmã Nilse, sua assistente virtual. Aqui está o resumo do dia de ontem.
Após a apresentação prossiga com o resumo.
Hoje é sexta feira
"""

print("Lista de tokens: ", lista_tokens)
print("Quantos tokens temos: ", len(lista_tokens))
print(f"Custo para o modelo {modelo} é de ${(len(lista_tokens)/1000) * 0.003}")

resposta_texto = cliente.chat.completions.create(
    messages=[
        {
            "role":"system",
            "content": promp_sistema
        },
        {
            "role":"user",
            "content":carrega("./teste.txt")
        }
    ],
    model=modelo
)

print(resposta_texto.choices[0].message.content)

response = cliente.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=resposta_texto.choices[0].message.content,
)

response.stream_to_file("resumo-do_dia.opus")