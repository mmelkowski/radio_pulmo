
def split_file(file_path, chunk_size):
    with open(file_path, 'rb') as myfile:
        chunk_number = 0
        while chunk := myfile.read(chunk_size):
            with open(f"{file_path}.part_{chunk_number}", 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunk_number += 1


if __name__ == "__main__":
    # Exemple d'utilisation :
    file_path = "/home/tylio/code/Project_radio_pulmo/code/radio_pulmo/models/EfficientNetB4_masked-Covid-19_masked-91.45.keras"  # Remplacez par le chemin de votre fichier
    chunk_size = 70 * 1024 * 1024  # Taille maximale d'une partie (ex. 70 Mo pour rester sous la limite de 100 Mo)
    split_file(file_path, chunk_size)
