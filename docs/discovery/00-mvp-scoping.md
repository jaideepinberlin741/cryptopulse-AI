### **🔧 Our MVP Definition for CryptoPulse AI**

**Team Context**  
*   **Team size:** 2
*   **Time:** 3 weeks
*   **Core skills:** Product Thinking, Agile PM, Programming (Python), EDA/Modeling. A very strong and well-aligned team!

**1️⃣ MVP Goal (User & Learning Focused)**

Our goal is to build a functional dashboard for the **"Structured Swing Trader" (Alex)** persona that provides a single, structured, and transparent market bias signal for BTC on a 4H timeframe. We want to learn if this structured signal, with its clear "why," increases a trader's decision-making confidence and is perceived as more valuable than fragmented, standalone indicators.

**2️⃣ Core Hypothesis to Validate**

We will be directly testing **"The Transparency Paradox" (H1)**: *Users say they want model explanations, but will they actually engage with them under pressure?*

Our MVP will confirm or reject the assumption that providing a clear, structured breakdown of the signal (the "why") builds more trust and confidence than just showing the output ("buy"/"sell").

**3️⃣ MVP Core Components (Must-Haves for 3 Weeks)**

To stay realistic and focused on validating the hypothesis, here is a lean "must-have" list:

*   **Feature 1: Core Signal Engine (Backend)**
    *   A Python backend (FastAPI would be great) that ingests BTC/USD 4H data.
    *   Calculates a single, primary signal: **Direction** (Bullish/Bearish) and **Probability** (e.g., 62%).
    *   Generates the **Alignment Matrix** data (Trend, Momentum, Volatility, News Bias) as this is *critical* for testing our transparency hypothesis.

*   **Feature 2: The Signal Dashboard (Frontend)**
    *   A simple web interface (using Lovable, as you mentioned) that displays the live signal for BTC.
    *   Clearly visualizes the Direction, Probability, the **Alignment Matrix**, and the **Volatility Risk Meter**. The risk meter is a high-impact feature for the "Sam" persona and a key differentiator.
    *   Includes a simple, one-click feedback mechanism to capture the **"Perceived Confidence Score" (Metric M9)** after a user views a signal. (e.g., "How confident do you feel now? 1-5 stars").

*   **Feature 3: Basic Data & Deployment**
    *   A reliable data source for BTC 4H price data.
    *   Deployment of the backend and frontend so you can test it with real users.

**4️⃣ Later Features (Should Have / Nice to Have)**

*   **Should Have:**
    *   **Signal History & Transparency:** The backtesting summary is a huge trust-builder.
    *   **"No-Trade" Signals:** A very powerful feature for building credibility.
    *   **User Accounts:** To track individual user feedback over time.

*   **Nice to Have:**
    *   Support for ETH and other assets.
    *   Push notifications.
    *   Advanced models (LSTM, etc.).
    *   In-app educational tooltips and content for the "Jordan" persona.

**5️⃣ Individual Learning Goals**

This is a great point for you and your teammate to discuss! Given your similar skill sets, you could:
*   **Specialize:** One of you could own the entire backend (data pipeline, API, model logic) while the other owns the frontend and user feedback mechanisms.
*   **Pair Up:** You could work together on all parts, which might be great for knowledge sharing on a new stack.

A possible learning goal could be: *"Mastering the deployment of a full-stack AI application, from the Python/FastAPI backend to a user-facing Lovable frontend within a 3-week sprint."*
