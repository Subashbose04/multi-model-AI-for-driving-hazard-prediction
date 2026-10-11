🛡️ Driving Hazard Prediction (DHPR)

An AI-powered application that analyzes driving-scene images and videos to identify visual objects and generate textual descriptions of potential road hazards.

The application combines AI model inference, computer vision, image and video processing, and an interactive Streamlit interface to present driving-scene analysis results.

📌 Project Overview

Driving Hazard Prediction aims to help analyze road scenes by processing dashcam images and videos.

Users can upload an image or video, run the analysis, and view the generated hazard description alongside detected objects highlighted with bounding boxes.

For video inputs, the application extracts selected frames and performs detailed analysis on keyframes to reduce repeated computationally expensive processing.

Key Objectives

- Analyze driving-scene images and videos.
- Generate textual descriptions of potential driving hazards.
- Display detected objects using bounding boxes.
- Process video frames efficiently using frame sampling.
- Provide an interactive interface for visualizing analysis results.

✨ Features

- Image Analysis: Upload an image and generate a hazard prediction.
- Video Analysis: Extract selected frames from uploaded videos for analysis.
- Object Detection Visualization: Display bounding boxes, object labels, and confidence scores.
- AI-Generated Descriptions: Present textual descriptions returned by the hazard prediction model.
- Keyframe Analysis: Perform detailed text generation on selected video frames.
- Annotated Video Output: Compile processed video frames into an animated GIF.
- Interactive Web Interface: Use Streamlit to upload media and view results.
- Processing-Time Display: Show the time taken to analyze images and frames.
- Command-Line Inference: Run predictions on individual images using a separate Python script.

🛠️ Technology Stack

Technology| Purpose
Python| Application logic and AI inference workflow
Streamlit| Interactive web interface
PyTorch| Deep learning framework imported by the inference script
OpenCV| Video reading and frame extraction
Pillow (PIL)| Image processing and bounding-box visualization
NumPy| Numerical computing library imported in the application
argparse| Command-line argument handling
tempfile| Temporary video file handling

Note: The internal implementation of the AI model is defined in "model.py", which must be included in the complete project. The exact model architecture and training procedure should be documented after verification.

⚙️ Project Pipeline

The application follows this workflow:

1. Input: The user uploads a driving-scene image or video.
2. Input Identification: The application identifies the input type.
3. Preprocessing:
   - Images are loaded and converted to RGB.
   - Videos are processed using OpenCV to extract selected frames.
4. Model Initialization: The application loads the "HazardPredictionModel".
5. AI Analysis:
   - Selected frames undergo full model analysis.
   - Other video frames undergo bounding-box detection, with the latest caption reused.
6. Visualization: Bounding boxes, object labels, confidence scores, and a short caption are added to the frames.
7. Result Processing: The application stores the predictions, annotated images, and processing times.
8. Output:
   - For images: Display the hazard description and annotated image.
   - For videos: Generate an animated GIF and display detailed results for selected keyframes.

Pipeline Diagram

        Driving Scene Input
                 |
                 v
        Image or Video Upload
                 |
                 v
          Input Processing
                 |
          +------+------+
          |             |
          v             v
      Image Input    Video Input
          |             |
          |        Frame Extraction
          |          (OpenCV)
          |             |
          +------+------+
                 |
                 v
          RGB Image Frames
                 |
                 v
        Load AI Prediction Model
                 |
                 v
          Analyze Video Frames
                 |
          +------+------+
          |             |
          v             v
      Keyframes     Other Frames
          |             |
          v             v
     Text Generation  Object Detection
     + Object         + Reused Caption
       Detection           |
          |                |
          +--------+-------+
                   |
                   v
        Draw Bounding Boxes
          and Object Labels
                   |
                   v
          Generate Results
                   |
            +------+------+
            |             |
            v             v
        Image Output   Video GIF
            |             |
            v             v
        Hazard Text   Keyframe Results
        + Annotated   + Annotated Frames
        Image

📂 Project Structure

The following is an example structure for organizing the project. Adjust filenames to match your actual repository.

Driving-Hazard-Prediction/
│
├── app.py                 # Streamlit application
├── inference.py           # Command-line image inference
├── model.py               # AI model implementation
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
│
├── models/                # Optional: model weights or assets
│
├── sample_images/         # Optional: sample input images
│
└── outputs/               # Optional: generated results

Important: The "model.py", "requirements.txt", and optional folders above are part of the suggested project organization; their contents and availability have not been verified from the two uploaded files. Rename "app-1.py" to "app.py" and "inference-1.py" to "inference.py" only if you choose to organize the files this way.

🚀 Installation and Setup

1. Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>

Navigate to the project directory:

cd Driving-Hazard-Prediction

2. Create a Virtual Environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

3. Install Dependencies

If a "requirements.txt" file is available:

pip install -r requirements.txt

Otherwise, install the required packages and any additional dependencies specified by your model implementation.

The supplied scripts import Streamlit, PyTorch, OpenCV, NumPy, and Pillow. The exact compatible package versions must be determined from the complete project.

4. Verify Model Files

Make sure the following are available:

- "model.py"
- Any model weights or configuration files required by "HazardPredictionModel"
- Any additional dependencies required by the model

The application cannot run successfully without its required model implementation and assets.

▶️ How to Run the Application

Option 1: Launch the Streamlit Application

Run:

streamlit run app.py

If you retain the original filename, use:

streamlit run app-1.py

Then:

1. Open the local application URL displayed in the terminal.
2. Upload a supported image or video.
3. Click Run Full Hazard Analysis.
4. Wait for the model to finish processing.
5. Review the hazard description and detected objects.

Option 2: Run Image Inference from the Terminal

To analyze a single image, run:

python inference.py --image road.jpg

Or, using the original filename:

python inference-1.py --image road.jpg

Replace "road.jpg" with the path to your input image.

The script prints the prediction and number of detected entities, draws up to three bounding boxes, and saves the annotated result as:

output_prediction.jpg

The inference script also accepts an optional device argument:

python inference.py --image road.jpg --device cpu

The selected device must be compatible with the model implementation and the installed dependencies.

🖼️ Supported Input Formats

Images

- JPG
- JPEG
- PNG
- BMP
- WEBP

Videos

- MP4
- AVI
- MOV
- MKV
- WEBM

The application attempts to extract up to 24 frames from each uploaded video.

🧠 Core Concepts

This project involves the following technical concepts:

- AI Model Inference: Using an available model to generate predictions from new input data.
- Computer Vision: Processing and analyzing image and video data.
- Object Detection: Representing detected objects using bounding boxes and labels.
- Image Processing: Converting image formats and annotating images.
- Video Frame Extraction: Selecting frames from video input for analysis.
- Keyframe Analysis: Performing detailed analysis on selected frames.
- Model Caching: Reusing an initialized model through Streamlit resource caching.
- Command-Line Interfaces: Passing image paths and device settings as command-line arguments.
- Performance Monitoring: Measuring processing times and displaying progress.

⚠️ Limitations

- Video analysis processes selected frames rather than every frame.
- Hazards appearing between sampled frames may not be detected.
- Captions reused between keyframes may not reflect changes in the scene.
- Processing time depends on the model, hardware, and input.
- Model accuracy and reliability must be evaluated using an appropriate test dataset.
- The application is an analysis prototype and should not be treated as a certified vehicle safety system.

🔮 Future Enhancements

Potential improvements include:

- Evaluate prediction quality using labeled driving-scene datasets.
- Improve frame sampling to capture brief hazards.
- Add object tracking across consecutive frames.
- Optimize inference performance.
- Evaluate compatible GPU acceleration.
- Add structured logging and improved error handling.
- Deploy the application for remote access.
- Develop a REST API for integrating predictions with other applications.
- Improve hazard evaluation using additional contextual information, where supported by the model.

🎯 Learning Outcomes

This project provides practical experience with:

- Python application development
- AI model integration and inference
- Computer vision
- Image and video processing
- Bounding-box visualization
- Streamlit application development
- Command-line tools
- Performance monitoring and optimization concepts

⚠️ Safety Disclaimer

This project is intended for educational and experimental purposes. Its predictions may be incomplete or incorrect and must not be relied upon as the sole basis for driving decisions or accident prevention.

👨‍💻 Author

Subash Bose C

Python | Artificial Intelligence | Machine Learning | Computer Vision

---

⭐ If you find this project interesting, feel free to explore the code and suggest improvements.
