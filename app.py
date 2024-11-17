import streamlit as st
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt

# Fungsi untuk konversi gambar ke format MNIST
def convert_to_mnist_format(img):
    # Ubah ukuran gambar menjadi 28x28 piksel
    img = img.resize((28, 28))
    # Konversi ke grayscale
    img = img.convert('L')
    # Konversi ke array numpy
    img_array = np.array(img)
    # Normalisasi ke rentang 0-1
    img_array = img_array / 255.0
    return img, img_array

# Fungsi untuk menampilkan array sebagai visualisasi
def display_pixel_array(array):
    fig, ax = plt.subplots()
    ax.imshow(array, cmap='gray')
    ax.axis('off')  # Hilangkan sumbu
    return fig

# Judul aplikasi
st.title("MNIST Converter App")
st.write("Imamge to  MNIST (28x28 grayscale).")

# Pilihan apakah pengguna ingin melakukan konversi batch
batch_mode = st.checkbox(" batch (batch convert)")

# Unggah file
if batch_mode:
    uploaded_files = st.file_uploader(
        "upload your image", type=["jpg", "png", "jpeg"], accept_multiple_files=True
    )
else:
    uploaded_files = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

if uploaded_files:
    if not batch_mode:  # Single image mode
        uploaded_files = [uploaded_files]

    # Loop melalui semua file yang diunggah
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Gambar Asli: {uploaded_file.name}", use_column_width=True)

        # Opsi auto-labeling
        label = st.text_input(f"Labelling {uploaded_file.name} (opsional):", value="")

        # Konversi ke format MNIST
        mnist_image, mnist_array = convert_to_mnist_format(image)

        # Tampilkan hasil konversi
        st.image(mnist_image, caption=f"Hasil MNIST (28x28 Grayscale): {uploaded_file.name}", use_column_width=True)

        # Visualisasi array menggunakan matplotlib
        st.write("Array Piksel Visualization:")
        st.pyplot(display_pixel_array(mnist_array))

        # Tampilkan array sebagai teks
        st.write("Array Gambar:")
        st.write(mnist_array)

        # Unduh gambar hasil konversi
        buf = io.BytesIO()
        mnist_image.save(buf, format="PNG")
        byte_data = buf.getvalue()
        st.download_button(
            label=f"Download {uploaded_file.name}",
            data=byte_data,
            file_name=f"mnist_{uploaded_file.name}",
            mime="image/png"
        )

        # Tampilkan label (jika ada)
        if label:
            st.success(f"Label for {uploaded_file.name}: {label}")

# Catatan tambahan
st.info("make sure your pictures have a good aspect ratio.")
