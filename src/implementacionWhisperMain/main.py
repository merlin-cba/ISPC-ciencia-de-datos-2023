import pywhatkit
import os
from modules.listen import Listener
from modules.talk import Talk
from modules.notes import Notes
from modules.chatgpt import Chat

def main():
    while True:

        listener = Listener()
        talker = Talk()
        notes = Notes()
        chatgpt = Chat()

        try:
            input = listener.listen()
            response = input.lower()

            if 'reproduce en youtube' in response:
                song = response.replace('reproduce', '')
                talker.talk(f"Reproduciendo {song}")
                pywhatkit.playonyt(song)

            if 'escribe' in response:
                name_new_note = notes.file_name()

                texto = response.replace('escribe', '')
                with open(name_new_note, 'w') as f:
                    f.write(f"{texto}")
                    f.close()
                    talker.talk(f"Escribí {texto}")

            if 'borrar notas' in response:
                folder_path = 'Archivos/Texto/'
                files = os.listdir(folder_path)
                i = 0
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    os.remove(file_path)
                    i = i+1
                talker.talk(f"Borré {i} notas")

            if 'lista de notas' in response:
                directorio = 'Archivos/Texto/'
                archivos = os.listdir(directorio)
                archivos_txt = [archivo for archivo in archivos if archivo.endswith('.txt')]
                talker.talk(archivos_txt)
            
            if 'chat' in response:
                input = response.replace('chat', '')
                request = chatgpt.chatgpt(input)
                talker.talk(request)


            if ('descansa' or 'descanza') in response:
                talker.talk('Procedo a descansar')
                break


        except Exception as e:
            print(f"Los siento no te entendí debido a este error: {e}")
            print(e)

if __name__ == '__main__':
    main()