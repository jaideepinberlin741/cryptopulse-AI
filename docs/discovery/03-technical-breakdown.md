### **Feature 1: Core Signal Engine (Backend)**
*This includes the logic for the Signal, the Alignment Matrix, and the Risk Meter.*

| Area | Description | Effort (S/M/C) | Risks & Dependencies |
| :--- | :--- | :--- | :--- |
| **Data / Model Pipeline** | **Setup Data Ingestion:** Create a Python script to fetch and clean BTC/USD 4H data from a reliable crypto exchange API. | **S** (2 days) | API rate limits; ensuring data quality and handling missing data points. |
| **Data / Model Pipeline** | **Feature Engineering:** Use `pandas-ta` to generate the core indicators for trend, momentum, and volatility (e.g., MAs, RSI, ATR). | **M** (3 days) | Selecting the right combination of indicators is experimental. Requires iteration. |
| **Data / Model Pipeline** | **Initial Model Training:** Train a baseline classification model (e.g., Scikit-learn's Logistic Regression or a simple LightGBM) to produce the `Direction` and `Probability` outputs. | **M** (3 days) | The initial model's predictive power might be low, but the goal is to have a working signal, not a perfect one. |
| **Backend** | **Create Signal API Endpoint:** Build a FastAPI endpoint (e.g., `/signal/btc`) that loads the model, processes the latest data, and returns a JSON payload with the full signal information. | **S** (2 days) | Depends on the model pipeline being complete. |

---

### **Feature 2: Streamlit Dashboard & User Feedback**
*This covers all the user-facing components.*

| Area | Description | Effort (S/M/C) | Risks & Dependencies |
| :--- | :--- | :--- | :--- |
| **Frontend** | **Build Core Dashboard UI:** Set up the main Streamlit application. Design the UI to clearly display the core signal, the Alignment Matrix, and the Volatility Risk Meter using Plotly or native components. | **S** (2 days) | Visual design needs to be clear and intuitive to avoid overwhelming the user. |
| **Frontend** | **Implement Feedback Mechanism:** Add the "Perceived Confidence Score" (PCS) widget, likely using buttons or a star rating, to the Streamlit UI. | **S** (1 day) | Trivial technical risk. |
| **Integration** | **API to Frontend Connection:** Write the logic within the Streamlit app to call your FastAPI backend, parse the JSON response, and dynamically update the dashboard components with the live data. | **S** (2 days) | **CRITICAL DEPENDENCY.** The frontend is blocked until the first version of the API is live. |
| **Analytics** | **Implement Basic Event Logging:** Add simple logging to the backend to track when the feedback endpoint is hit. This provides the raw data for our `Perceived Confidence Score` KPI. | **S** (1 day) | Needs to be in place to measure the Outcome Goal. |

---

### **Feature 3: Deployment & Testing**

| Area | Description | Effort (S/M/C) | Risks & Dependencies |
| :--- | :--- | :--- | :--- |
| **Integration** | **Dockerize Application:** Create Dockerfiles for both the FastAPI backend and the Streamlit frontend to ensure a consistent environment for development and deployment. | **S** (2 days) | Can be tricky if you're new to Docker networking between containers. |
| **Integration** | **Deploy to Cloud:** Deploy the containerized application to a simple cloud host so it's accessible to your team and a small group of test users. | **S** (2 days) | Depends on Dockerization being complete. |

---

### **Summary & Next Steps**

*   **Total Estimated Effort:** The tasks sum up to roughly **18 person-days**. For a team of two, this is very achievable within a 3-week (30 person-days) sprint, leaving a healthy buffer for testing, bug fixes, and planning.
*   **Critical Path:** The most significant dependency is **Data/Model Pipeline → Backend API → Frontend Integration**. This is the core sequence you'll need to follow.