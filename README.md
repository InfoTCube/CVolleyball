# CVolleyball - Automated Volleyball Reception Analysis System

## 📌 Overview
Goal of this program is to automate the technical assessment of volleyball reception by analyzing video footage. Its core function is to objectively evaluate the quality and effectiveness of a player's reception based on computer vision analysis of ball trajectory, player positioning, and timing.

## 🏐 Ball Tracking Implementation
The system's ball tracking is powered by a specialized YOLO  model specifically fine-tuned for detecting volleyballs. The approach was selected for its strong balance of high precision and recall, ensuring reliable detection across diverse game conditions like lighting, camera angles, and fast ball movement.

Source Model: The ball detection model (yolo11n-volleyball.pt) originates from the [jadidimohammad/volleyball-tracking](https://github.com/jadidimohammad/volleyball-tracking) repository on GitHub. This model was trained on the extensive "Volleyvision" dataset (~17,000 training images), making it highly accurate for volleyball-specific scenarios.

Why This Method?: Utilizing a pre-trained, sport-specific model provides a robust and accurate foundation for tracking, which is more effective and efficient than building a general-purpose tracker from scratch. This allows the project to focus on the higher-level analysis logic for reception grading.

#### Example
![Ball-tracking-video](examples/ball_tracking_output.gif)

## 🚀 Getting Started
#### Prerequisites
Ensure you have Python installed, then install the required dependencies from requirements.txt.

#### Installation
Clone or download this repository.

#### Install dependencies:

```bash
pip install -r requirements.txt
```
Ensure your trained model file (e.g., yolo11n-volleyball.pt) is in the correct project directory.

### Usage
Run the main analysis script, specifying the path to your volleyball video.

```bash
python src/ball_tracking.py
```

## 🎯 Key Features (in the future)
- Automated Reception Scoring: Analyzes video to provide a quantitative score for each reception.
- Robust Ball Tracking: Leverages a fine-tuned YOLO model for consistent ball detection.
- Trajectory & Motion Analysis: Calculates ball speed, direction, and path for in-depth assessment.
- Real-Time/Post-Processing: Can analyze video in real-time (from a camera) or process recorded footage.

## 🔮 Future Enhancements
Potential future improvements could include player action recognition, team formation analysis, and generating detailed performance reports.