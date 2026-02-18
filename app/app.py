# =============================================================================
# Gaming Engagement Prediction - Production Streamlit Application
# =============================================================================
"""
Gaming Engagement Predictor
---------------------------
A machine learning web application that predicts player engagement levels
(Low, Medium, High) based on gameplay behavior and player characteristics.

Author: Your Name
Date: 2024
Version: 1.0.0
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import pandas as pd
import streamlit as st
import yaml
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration Management
# -----------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Application configuration settings."""
    
    # Page settings
    page_title: str = "Gaming Engagement Predictor"
    page_icon: str = "🎮"
    layout: str = "centered"
    
    # Paths
    base_dir: Path = Path(__file__).resolve().parent
    model_dir: Path = base_dir.parent / "models"
    model_filename: str = "xgboost_pipeline.joblib"
    config_dir: Path = base_dir.parent / "config"
    
    # Model settings
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["Low", "Medium", "High"]
        self.model_path = self.model_dir / self.model_filename
        self.config_path = self.config_dir / "config.yaml"


class ConfigLoader:
    """Load and validate configuration files."""
    
    @staticmethod
    def load_config(config_path: Path) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            return {}


# -----------------------------------------------------------------------------
# Model Management
# -----------------------------------------------------------------------------

class ModelLoader:
    """Handle model loading with caching and error handling."""
    
    def __init__(self, config: AppConfig):
        """
        Initialize ModelLoader.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.model = None
        
    @st.cache_resource(show_spinner="Loading model...")
    def load_model(_self) -> Optional[object]:
        """
        Load the trained model pipeline.
        
        Returns:
            Loaded model or None if loading fails
        """
        try:
            if not _self.config.model_path.exists():
                logger.error(f"Model file not found: {_self.config.model_path}")
                st.error(f"""
                    Model not found at: {_self.config.model_path}
                    
                    Please ensure:
                    1. The model has been trained
                    2. The model file exists in the correct directory
                    3. File permissions are correct
                """)
                return None
                
            model = joblib.load(_self.config.model_path)
            logger.info(f"Model loaded successfully from {_self.config.model_path}")
            
            # Verify model has required methods
            if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
                logger.error("Loaded model doesn't have required methods")
                st.error("Invalid model format. Please retrain the model.")
                return None
                
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            st.error(f"""
                Error loading model: {str(e)}
                
                Troubleshooting steps:
                1. Check if model was trained with correct dependencies
                2. Verify model file integrity
                3. Check application logs for details
            """)
            return None


# -----------------------------------------------------------------------------
# Input Validation
# -----------------------------------------------------------------------------

class InputValidator:
    """Validate and process user inputs."""
    
    @staticmethod
    def validate_numeric_input(value: float, min_val: float, max_val: float, 
                               name: str) -> Tuple[bool, str]:
        """
        Validate numeric input ranges.
        
        Args:
            value: Input value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Field name for error message
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if value < min_val or value > max_val:
            return False, f"{name} must be between {min_val} and {max_val}"
        return True, ""
    
    @staticmethod
    def prepare_input_data(inputs: Dict) -> pd.DataFrame:
        """
        Prepare input data for model prediction.
        
        Args:
            inputs: Dictionary of input features
            
        Returns:
            DataFrame formatted for model input
        """
        return pd.DataFrame([inputs])


# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

class UIComponents:
    """Manage Streamlit UI components and styling."""
    
    @staticmethod
    def apply_custom_styling():
        """Apply custom CSS styling to the app."""
        st.markdown("""
        <style>
            /* Global Styles */
            body {
                background-color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            /* Main Title Styling - Large and Attractive */
            .main-title {
                text-align: center;
                font-size: 3.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 0.5rem;
                padding-top: 1rem;
                letter-spacing: -0.02em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Game Controller Animation */
            .title-emoji {
                display: inline-block;
                animation: bounce 2s infinite;
            }
            
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }
            
            .subtitle {
                text-align: center;
                font-size: 1.2rem;
                color: #475569;
                margin-bottom: 2rem;
                font-weight: 400;
                max-width: 800px;
                margin-left: auto;
                margin-right: auto;
                line-height: 1.6;
            }
            
            /* Dark Section Headers */
            .section-header {
                font-size: 1.5rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 1.5rem;
                padding: 1rem 1.5rem;
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 12px;
                color: white;
                letter-spacing: -0.01em;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                border-left: 4px solid #667eea;
            }
            
            .prediction-header {
                font-size: 1.5rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 1.5rem;
                padding: 1rem 1.5rem;
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 12px;
                color: white;
                letter-spacing: -0.01em;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                border-left: 4px solid #fbbf24;
            }
            
            /* Card Containers */
            .card {
                border: 1px solid #e2e8f0;
                border-radius: 1rem;
                padding: 1.5rem;
                background: white;
                box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
                margin-bottom: 1.5rem;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
            }
            
            /* Button Styling */
            .stButton > button {
                width: 100%;
                height: 3.5rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 700;
                font-size: 1.2rem;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-top: 1rem;
                box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.4);
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 20px 25px -5px rgba(102, 126, 234, 0.5);
            }
            
            /* Result Boxes - Clean and Compact */
            .result-box {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 1.25rem;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }
            
            .result-header {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.5rem;
            }
            
            .result-icon {
                font-size: 1.8rem;
            }
            
            .result-title {
                font-size: 1.2rem;
                font-weight: 600;
            }
            
            .result-title.high { color: #059669; }
            .result-title.medium { color: #d97706; }
            .result-title.low { color: #dc2626; }
            
            .result-description {
                color: #4b5563;
                font-size: 0.95rem;
                line-height: 1.4;
                margin: 0;
                padding-left: 2.5rem;
            }
            
            /* NEW: Modern Confidence Score Display - Card Style */
            .confidence-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 1rem;
                margin: 1.5rem 0;
            }
            
            .confidence-card {
                background: white;
                border-radius: 12px;
                padding: 1.25rem 0.75rem;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                border: 1px solid #e2e8f0;
                transition: transform 0.2s;
            }
            
            .confidence-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }
            
            .confidence-card.high {
                border-top: 4px solid #059669;
            }
            
            .confidence-card.medium {
                border-top: 4px solid #d97706;
            }
            
            .confidence-card.low {
                border-top: 4px solid #dc2626;
            }
            
            .confidence-value {
                font-size: 2.2rem;
                font-weight: 700;
                line-height: 1.2;
                margin-bottom: 0.25rem;
            }
            
            .confidence-value.high { color: #059669; }
            .confidence-value.medium { color: #d97706; }
            .confidence-value.low { color: #dc2626; }
            
            .confidence-label {
                font-size: 0.9rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #64748b;
                margin-bottom: 0.25rem;
            }
            
            .confidence-subtext {
                font-size: 0.75rem;
                color: #94a3b8;
            }
            
            /* Footer Styling */
            .footer {
                text-align: center;
                padding: 2rem 0;
                color: #64748b;
                font-size: 0.9rem;
                border-top: 2px solid #e2e8f0;
                margin-top: 2rem;
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 12px;
            }
            
            /* Tooltips */
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: help;
            }
            
            .tooltip .tooltiptext {
                visibility: hidden;
                background-color: #1e293b;
                color: #fff;
                text-align: center;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 0.875rem;
                white-space: nowrap;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            
            /* Input Labels */
            .input-label {
                font-weight: 600;
                color: #1e293b;
                margin-bottom: 0.25rem;
            }
            
            /* Metric Cards */
            .metric-card {
                background: white;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border: 1px solid #e2e8f0;
                text-align: center;
                transition: all 0.2s;
            }
            
            .metric-card:hover {
                transform: scale(1.02);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            /* Sidebar Styling */
            .css-1d391kg {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            }
            
            .sidebar-content {
                color: white;
            }
            
            /* Stats Container */
            .stats-container {
                display: flex;
                justify-content: space-around;
                margin: 2rem 0;
                padding: 1rem;
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 12px;
            }
            
            .stat-item {
                text-align: center;
            }
            
            .stat-value {
                font-size: 2rem;
                font-weight: 700;
                color: #667eea;
            }
            
            .stat-label {
                color: #64748b;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_confidence_scores(probabilities: List[float], class_names: List[str]):
        """
        Display prediction probabilities in a modern, visually appealing format.
        
        Args:
            probabilities: List of probability values
            class_names: List of class names
        """
        # Create a container for the confidence scores
        st.markdown('<div style="margin: 1.5rem 0;">', unsafe_allow_html=True)
        st.markdown("#### 📊 Confidence Scores")
        
        # Create 3-column grid for confidence cards
        cols = st.columns(3)
        
        for idx, (col, label, prob) in enumerate(zip(cols, class_names, probabilities)):
            percentage = f"{prob:.1%}"
            color_class = label.lower()
            
            # Determine subtext based on probability
            if prob >= 0.7:
                subtext = "High confidence"
            elif prob >= 0.4:
                subtext = "Moderate confidence"
            else:
                subtext = "Low confidence"
            
            with col:
                st.markdown(f"""
                <div class="confidence-card {color_class}">
                    <div class="confidence-value {color_class}">{percentage}</div>
                    <div class="confidence-label">{label}</div>
                    <div class="confidence-subtext">{subtext}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

def main():
    """Main application entry point."""
    
    # Initialize configuration
    config = AppConfig()
    
    # Page configuration
    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout=config.layout,
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom styling
    UIComponents.apply_custom_styling()
    
    # Load configuration
    config_data = ConfigLoader.load_config(config.config_path)
    
    # Title section with larger font and animation
    st.markdown("""
    <div style='text-align: center;'>
        <span class='title-emoji' style='font-size: 4rem;'>🎮</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='main-title'>Gaming Engagement Level Prediction</div>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class='subtitle'>
        🚀 Predict player engagement levels using advanced machine learning based on 
        gameplay behavior and player characteristics. Get instant insights with 
        confidence scores!
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>🎯</div>
            <div style='font-weight: 600;'>3 Levels</div>
            <div style='color: #64748b; font-size: 0.8rem;'>Low/Medium/High</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>📊</div>
            <div style='font-weight: 600;'>11 Features</div>
            <div style='color: #64748b; font-size: 0.8rem;'>Player Attributes</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>⚡</div>
            <div style='font-weight: 600;'>Real-time</div>
            <div style='color: #64748b; font-size: 0.8rem;'>Instant Results</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>🎨</div>
            <div style='font-weight: 600;'>Interactive</div>
            <div style='color: #64748b; font-size: 0.8rem;'>Easy to Use</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar with information (removed model performance section)
    with st.sidebar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
            <h3 style='color: white; margin: 0;'>ℹ️ About</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        This application uses an **XGBoost classifier** to predict player 
        engagement levels based on various gameplay metrics.
        
        ### Features Used:
        - 👤 **Demographics** (Age, Gender, Location)
        - ⏱️ **Gameplay Metrics** (Play time, Sessions, Duration)
        - 📈 **Game Progress** (Level, Achievements)
        - 🎮 **Game Preferences** (Genre, Difficulty)
        
        ### Engagement Levels:
        - 🔴 **Low**: Casual players with minimal engagement
        - 🟡 **Medium**: Regular players with moderate engagement
        - 🟢 **High**: Dedicated players with high engagement
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. Fill in player information
        2. Click predict button
        3. View instant results with confidence scores
        """)
    
    # Load model
    model_loader = ModelLoader(config)
    model = model_loader.load_model()
    
    if model is None:
        st.stop()
    
    # Input form with dark header
    with st.container():
        st.markdown("<div class='section-header'>📊 Player Information</div>", 
                   unsafe_allow_html=True)
        
        with st.form(key="prediction_form", clear_on_submit=False):
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input(
                    "Age",
                    min_value=10,
                    max_value=100,
                    value=25,
                    step=1,
                    help="Player's age in years"
                )
                
                play_time = st.number_input(
                    "Total Play Time (Hours)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=100.0,
                    step=10.0,
                    help="Total hours played across all sessions"
                )
                
                sessions_per_week = st.number_input(
                    "Sessions Per Week",
                    min_value=0,
                    max_value=50,
                    value=5,
                    step=1,
                    help="Average number of gaming sessions per week"
                )
                
                avg_session_duration = st.number_input(
                    "Average Session Duration (Minutes)",
                    min_value=1,
                    max_value=480,
                    value=30,
                    step=5,
                    help="Average length of each gaming session"
                )
                
                player_level = st.number_input(
                    "Player Level",
                    min_value=1,
                    max_value=200,
                    value=10,
                    step=1,
                    help="Current player level in the game"
                )
            
            with col2:
                achievements = st.number_input(
                    "Achievements Unlocked",
                    min_value=0,
                    max_value=500,
                    value=10,
                    step=5,
                    help="Number of achievements earned"
                )
                
                in_game_purchases = st.selectbox(
                    "In-Game Purchases",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Whether player has made in-game purchases"
                )
                
                gender = st.selectbox(
                    "Gender",
                    options=["Male", "Female", "Other", "Prefer not to say"],
                    help="Player's gender"
                )
                
                location = st.selectbox(
                    "Location",
                    options=["USA", "Europe", "Asia", "South America", "Africa", "Oceania", "Other"],
                    help="Player's geographic region"
                )
                
                game_genre = st.selectbox(
                    "Game Genre",
                    options=["Action", "RPG", "Sports", "Strategy", "Adventure", "Simulation", "Puzzle"],
                    help="Preferred game genre"
                )
                
                game_difficulty = st.selectbox(
                    "Game Difficulty",
                    options=["Easy", "Medium", "Hard", "Expert"],
                    help="Preferred difficulty level"
                )
            
            # Submit button
            submitted = st.form_submit_button(
                label="🎯 Predict Engagement Level",
                use_container_width=True
            )
    
    # Prediction and results with dark header
    if submitted:
        try:
            # Prepare input data
            input_data = {
                "Age": age,
                "PlayTimeHours": play_time,
                "SessionsPerWeek": sessions_per_week,
                "AvgSessionDurationMinutes": avg_session_duration,
                "PlayerLevel": player_level,
                "AchievementsUnlocked": achievements,
                "InGamePurchases": in_game_purchases,
                "Gender": gender,
                "Location": location,
                "GameGenre": game_genre,
                "GameDifficulty": game_difficulty
            }
            
            # Validate inputs
            validator = InputValidator()
            input_df = validator.prepare_input_data(input_data)
            
            # Make prediction
            with st.spinner("🔮 Analyzing player data..."):
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
            
            # Display results with dark header
            st.markdown("<div class='prediction-header'>🎯 Prediction Results</div>", 
                       unsafe_allow_html=True)
            
            engagement_label = config.class_names[prediction]
            
            # Professional result boxes - clean and compact
            if engagement_label == "High":
                st.markdown("""
                <div class="result-box">
                    <div class="result-header">
                        <span class="result-icon">🏆</span>
                        <span class="result-title high">High Engagement Player</span>
                    </div>
                    <p class="result-description">
                        Player shows characteristics of highly engaged gamers with strong retention potential.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif engagement_label == "Medium":
                st.markdown("""
                <div class="result-box">
                    <div class="result-header">
                        <span class="result-icon">📊</span>
                        <span class="result-title medium">Medium Engagement Player</span>
                    </div>
                    <p class="result-description">
                        Player demonstrates moderate engagement with growth potential.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box">
                    <div class="result-header">
                        <span class="result-icon">⚠️</span>
                        <span class="result-title low">Low Engagement Player</span>
                    </div>
                    <p class="result-description">
                        Player shows low engagement patterns. Consider retention strategies.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display confidence scores in modern card format
            UIComponents.display_confidence_scores(probabilities, config.class_names)
            
            # UPDATED: Feature importance explanation with clean, simple text
            with st.expander("🔍 How the Model Determines Engagement"):
                st.markdown("""
                The model primarily evaluates **player activity patterns** to determine engagement level.

                - **Sessions per Week** and **Average Session Duration** are the strongest indicators, 
                  meaning consistent and longer gameplay signals higher engagement.

                - **Game Genre, Location, and Difficulty** have moderate influence, 
                  reflecting how player preferences shape engagement behavior.

                - **Achievements, Player Level, and Total Play Time** contribute to a lesser extent.

                - **Demographics and Purchases** have minimal impact on the final prediction.

                Overall, the model prioritizes consistent gameplay behavior over demographics or spending.
                """)
            
            # Log prediction
            logger.info(f"Prediction made: {engagement_label} - Probabilities: {probabilities}")
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            st.error(f"""
                An error occurred during prediction: {str(e)}
                
                Please try again or contact support if the issue persists.
            """)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <div style='display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;'>
            <span>🎮 Gaming Engagement ML Project</span>
            <span>•</span>
            <span>Version 1.0.0</span>
            <span>•</span>
            <span>© 2024</span>
        </div>
        <p style='color: #64748b; font-size: 0.8rem;'>
            Built with Streamlit • XGBoost Classifier • Production Ready
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()