# Bone-Fracture-Detection
A Streamlit app using a TensorFlow model to detect bone fractures from X-ray images. Upload an X-ray to get a quick prediction if a fracture is present, helping medical professionals with fast and accurate preliminary diagnosis with an accuracy of 89.10 %

## Features

- Upload X-ray images and get predictions: **Fractured** or **Not Fractured**  
- Displays confidence scores for the predictions  
- Simple, easy-to-use web interface with Streamlit  
- Model trained with TensorFlow/Keras  

---

## Getting Started

### Prerequisites

- Python 3.8 or above  
- GPU recommended but not required  

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/AlbinJosephG/Bone-Fracture-Detection.git
    cd Bone-Fracture-Detection
    ```

2. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```bash
    streamlit run web_app/streamlit_app.py
    ```


## Usage

- Open the app in your browser (usually at `http://localhost:8501/`)  
- Upload an X-ray image (JPG, PNG)  
- View the fracture detection result along with confidence score  

## Project Structure

- data/
  - train/
    - fractured/
    - not fractured/
  - val/
    - fractured/
    - not fractured/
- inference/
  - predict.py
- model/
  - fracture_classifier.h5
  - train_model.py
- web_app/
  - streamlit_app.py
- requirements.txt
- README.md


## Notes

- The model file `fracture_classifier.h5` is about 75 MB. For better handling of large files, consider using [Git Large File Storage (Git LFS)](https://git-lfs.github.com/).  
- This app is for educational/demo purposes and not for clinical use.  

## License

This project is licensed under the MIT License.

---

## Contact

For questions or collaboration, please open an issue or contact me at itsalbinjoseph@gmail.com
