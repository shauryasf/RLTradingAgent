"""
Main Streamlit app entry point for deployment
"""
import streamlit as st

# MUST be first - configure page settings
st.set_page_config(
    page_title="RL Trading Agent",
    page_icon="📈",
    layout="wide"
)

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Now import the main app components with graceful fallbacks
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    PPO = None
    SB3_AVAILABLE = False
    st.warning("⚠️ Stable Baselines3 not available. Running in demo mode with simulated predictions.")

try:
    from src.envs import SingleStockTradingEnv
    from src.data.data_loader import DataLoader
    from src.agents.PPOAgent import TradingPPOAgent
    from src.inference.inference import TradingInferenceEngine
    from src.utils.metrics import PerformanceMetrics
    FULL_FEATURES = True
except ImportError as e:
    try:
        from src.envs.simple_env import SimpleTradingEnv as SingleStockTradingEnv
        st.info("🔧 Using simplified environment due to missing dependencies")
        FULL_FEATURES = False
        DataLoader = None
        TradingPPOAgent = None
        TradingInferenceEngine = None
        PerformanceMetrics = None
    except ImportError:
        st.error(f"Core import error: {e}")
        st.stop()

# --------------------------
# Model Management Section
# --------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"  # models/ at root level

class ModelManager:
    """Handles model loading and management for the UI"""
    
    def __init__(self):
        try:
            if FULL_FEATURES:
                self.data_loader = DataLoader()
                self.inference_engine = TradingInferenceEngine()
            else:
                self.data_loader = None
                self.inference_engine = None
                st.info("🎮 Running in simplified mode - some features may be limited")
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            self.data_loader = None
            self.inference_engine = None
    
    def get_available_tickers(self):
        """Get list of available model tickers"""
        if not MODELS_DIR.exists():
            return []
        return [f.stem for f in MODELS_DIR.glob("*.zip")]
    
    def load_model(self, ticker, env):
        """Load model from models/{ticker}.zip"""
        if not SB3_AVAILABLE:
            st.warning(f"⚠️ Cannot load model for {ticker} - Stable Baselines3 not available")
            return None
            
        model_path = MODELS_DIR / f"{ticker}.zip"

        if not model_path.exists():
            st.error(f"No model found for {ticker}!")
            return None

        try:
            model = PPO.load(model_path, env=env)
            st.success(f"Loaded model for {ticker}")
            return model

        except Exception as e:
            st.error(f"Error loading model for {ticker}: {str(e)}")
            return None

    def predict_action(self, model, env, portfolio_value, num_stock_shares):
        """Predict next action using the model"""
        try:
            return self.inference_engine.predict_action(
                model, env, portfolio_value, num_stock_shares
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return 0

# Initialize model manager
@st.cache_resource
def get_model_manager():
    return ModelManager()

model_manager = get_model_manager()

# --------------------------
# Streamlit UI
# --------------------------

st.title("📈 Single-Stock RL Trading Agent")
st.markdown("*Reinforcement Learning-powered stock trading recommendations*")

# Get tickers
TICKERS = model_manager.get_available_tickers()

if not TICKERS:
    st.error("No trained models found! Please ensure model files are in the models/ directory.")
    st.stop()

# Main interface
st.header("Trading Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    selected_ticker = st.selectbox("Select stock ticker", TICKERS)

with col2:
    num_stock_shares = st.number_input("Number of shares", 
                                       min_value=0, 
                                       value=0,
                                       step=1)

with col3:
    portfolio_value = st.number_input("Portfolio Value ($)", 
                                      min_value=0.0, 
                                      value=10000.0,
                                      step=100.0)

# Date
prediction_date = datetime.today()

# Update environment when ticker changes
if selected_ticker:
    with st.spinner(f"Loading data for {selected_ticker}..."):
        try:
            if 'current_ticker' not in st.session_state or st.session_state.current_ticker != selected_ticker:
                # Capture warnings from environment creation
                import io
                from contextlib import redirect_stdout, redirect_stderr
                
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    st.session_state.env = SingleStockTradingEnv(
                        ticker=selected_ticker, 
                        start_date="2020-01-01", 
                        end_date=prediction_date.strftime("%Y-%m-%d")
                    )
                
                # Check for dummy data warnings
                output = stdout_capture.getvalue()
                error_output = stderr_capture.getvalue()
                
                if "dummy data" in output.lower() or "fallback" in output.lower():
                    st.warning(f"⚠️ Using simulated data for {selected_ticker} due to API limitations. Predictions may not reflect real market conditions.")
                
                if "failed download" in output.lower() or "rate limited" in output.lower():
                    st.info("💡 Experiencing temporary data access issues. Using fallback data for demonstration.")
                
                st.session_state.current_ticker = selected_ticker

            # Load model when ticker changes
            model = model_manager.load_model(selected_ticker, st.session_state.env)
            if model:
                st.session_state.loaded_model = model

        except Exception as e:
            st.error(f"Error setting up environment: {e}")
            st.info("This might be due to API rate limits. Try again in a few moments.")

    # Historical Data
    if 'env' in st.session_state:
        st.header(f"📊 {selected_ticker} Historical Trends")
        
        try:
            data = st.session_state.env.df['close']
            st.line_chart(data)
            
            # Show basic stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${data.iloc[-1]:.2f}")
            with col2:
                st.metric("30-day Change", f"{((data.iloc[-1] / data.iloc[-30] - 1) * 100):.2f}%")
            with col3:
                st.metric("90-day High", f"${data.tail(90).max():.2f}")
            with col4:
                st.metric("90-day Low", f"${data.tail(90).min():.2f}")
                
        except Exception as e:
            st.warning(f"Unable to display historical data: {e}")

    # Prediction Section
    st.header("🤖 Trading Recommendation")
    
    if st.button("Generate Prediction", type="primary"):
        if not SB3_AVAILABLE:
            # Demo mode - generate fake predictions
            st.info("🎮 Demo Mode: Generating simulated prediction")
            import random
            random.seed(42)  # For consistent demo results
            demo_actions = ["BUY", "SELL", "HOLD"]
            demo_action = random.choice(demo_actions)
            demo_quantity = random.randint(1, 50) if demo_action != "HOLD" else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Recommended Action (Demo)", demo_action)
            with col2:
                st.metric("Recommended Quantity (Demo)", demo_quantity)
            st.info("💡 This is a simulated prediction for demonstration purposes.")
            
        elif 'loaded_model' not in st.session_state:
            st.error("Model failed to load! Please try again.")
        else:
            try:
                with st.spinner("Generating recommendation..."):
                    action = model_manager.predict_action(
                        st.session_state.loaded_model, 
                        st.session_state.env, 
                        portfolio_value, 
                        num_stock_shares
                    )

                    recommendation = "BUY" if action > 0 else ("SELL" if action < 0 else "HOLD")
                    
                    st.success(f"Prediction generated for {selected_ticker}")

                    col1, col2 = st.columns(2)
                    with col1:
                        color = "normal" if recommendation == "HOLD" else ("inverse" if recommendation == "SELL" else "off")
                        st.metric("Recommended Action", recommendation)
                    with col2:
                        st.metric("Recommended Quantity", abs(action))
                        
                    # Additional info
                    st.info(f"💡 Recommendation generated on {prediction_date.strftime('%Y-%m-%d')} based on current market conditions.")
                    
            except Exception as e:
                st.error(f"Error generating prediction: {e}")

# Footer
st.markdown("---")
st.markdown("*Powered by Stable Baselines3, FinRL, and Streamlit* 🚀")
