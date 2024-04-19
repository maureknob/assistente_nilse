from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
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




# image= Image.open("./tami.png")
# width, height = 256, 256
# image = image.resize((width, height))

# byte_stream = BytesIO()
# image.save(byte_stream, format='PNG')
# byte_array = byte_stream.getvalue()

# resposta_imagem = cliente.images.generate(
#     model="dall-e-3",
#     prompt="emergindo do esconderijo da escuridão, o terror se desenrola diante de vocês. É um dragão. Sua aparência é tão apavorante quanto a criatura mais horrenda dos seus pesadelos mais escuros. Seus olhos, faíscas ferozes, são duas estrelas cadentes amaldiçoadas, destilando desespero e destruição. Suas escamas, grandes placas de metal enegrecido, estão dispostas sobre seu corpo com uma precisão que desafia a própria natureza, formando uma armadura impenetrável. O monstro se inclina, e vocês veem suas asas se abrirem, imensas. São tão vastas que poderiam engolir a lua. E quando bate suas asas, vocês são atingidos por um vento que é mais escaldante do que qualquer aridez encontrada neste deserto. Seus chifres torcidos como ramos antigos são cobertos de cicatrizes de batalhas incontáveis. De sua boca, goteja um líquido viscoso, como lava a escorrer de um vulcão em erupção, derretendo a areia abaixo de suas mandíbulas poderosas que são repletas de presas afiadas como espadas. O dragão rosna, uma cacofonia perturbadora que estremece o chão e arrepia seus corações valentes. É um grito que não promete nada além de morte. Vocês, aventureiros, estão de frente para esse ser apocalíptico. Seu destino, o que quer que seja, aguarda nas faíscas ardentes dos olhos da fera.",
#     n=1,
#     size="1024x1024",
#     quality="standard",
# )

# print(resposta_imagem.data[0].url)