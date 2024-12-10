import streamlit as st


def merge_files(output_file_path, parts):
    with open(output_file_path, 'wb') as output_file:
        for part in parts:
            with open(part, 'rb') as part_file:
                output_file.write(part_file.read())



if __name__ == "__main__":
    # Exemple d'utilisation :
    parts = [
        "../../../models/EfficientNetB4_masked-Covid-19_masked-91.45.keras.part_0",
        "../../../models/EfficientNetB4_masked-Covid-19_masked-91.45.keras.part_1",
        "../../../models/EfficientNetB4_masked-Covid-19_masked-91.45.keras.part_2"
    ]
    output_file_path = "EfficientNetB4_masked-Covid-19_masked-91.45.keras"
    merge_files(output_file_path, parts)

