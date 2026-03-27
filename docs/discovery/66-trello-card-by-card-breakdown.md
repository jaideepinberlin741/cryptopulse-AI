### **List: Epic A: Backend**

**Card 1 of 7**
**Title:** `[A.1] Setup data ingestion script`
**Description:**
```
**User Story:** As a developer, I need to automatically fetch the latest BTC/USD 4H data so that the model always has fresh information.
---
**Acceptance Criteria:**
- Script successfully fetches and saves BTC 4H data from an exchange API.
- Data is saved locally to a CSV or Parquet file.
---
**Estimate:** 1 day
**Owner:** Member 1
**Dependencies:** None
```

**Card 2 of 7**
**Title:** `[A.2] Engineer features for model`
**Description:**
```
**User Story:** As a developer, I need to automatically fetch the latest BTC/USD 4H data so that the model always has fresh information.
---
**Acceptance Criteria:**
- Script loads raw data from the local file.
- Outputs a feature set including MAs, RSI, and ATR using `pandas-ta`.
---
**Estimate:** 1 day
**Owner:** Member 1
**Dependencies:** Task A.1
```

**Card 3 of 7**
**Title:** `[A.3] Train baseline classification model`
**Description:**
```
**User Story:** As a developer, I need to automatically fetch the latest BTC/USD 4H data so that the model always has fresh information.
---
**Acceptance Criteria:**
- Script trains a `LightGBM` model and saves the trained artifact (e.g., .pkl).
- Achieves a baseline accuracy > 55% on a held-out test set.
---
**Estimate:** 2 days
**Owner:** Member 1
**Dependencies:** Task A.2
```

**Card 4 of 7**
**Title:** `[A.4] Create FastAPI app structure`
**Description:**
```
**User Story:** As a developer, I want to expose the complete signal via a single API endpoint so that the frontend can easily consume it.
---
**Acceptance Criteria:**
- Basic "Hello World" FastAPI server is running locally.
- Project structure for `api/` routes is in place.
---
**Estimate:** 0.5 days
**Owner:** Member 1
**Dependencies:** None
```

**Card 5 of 7**
**Title:** `[A.5] Build signal generation logic`
**Description:**
```
**User Story:** As a developer, I want to expose the complete signal via a single API endpoint so that the frontend can easily consume it.
---
**Acceptance Criteria:**
- A Python function takes the latest data, loads the trained model, and generates the full JSON output (Direction, Probability, Alignment Matrix, Risk).
---
**Estimate:** 1.5 days
**Owner:** Member 1
**Dependencies:** Task A.3
```

**Card 6 of 7**
**Title:** `[A.6] Implement the /signal/btc endpoint`
**Description:**
```
**User Story:** As a developer, I want to expose the complete signal via a single API endpoint so that the frontend can easily consume it.
---
**Acceptance Criteria:**
- A `GET /signal/btc` endpoint successfully calls the signal logic and returns the correct JSON payload to the browser or Postman.
---
**Estimate:** 1 day
**Owner:** Member 1
**Dependencies:** Task A.5
```

**Card 7 of 7**
**Title:** `[A.7] Add basic analytics logging`
**Description:**
```
**User Story:** As a developer, I want to expose the complete signal via a single API endpoint so that the frontend can easily consume it.
---
**Acceptance Criteria:**
- A new endpoint (`POST /feedback`) is created.
- When this endpoint is hit with a score, it logs the score and a timestamp to the console or a local file.
---
**Estimate:** 1 day
**Owner:** Member 1
**Dependencies:** None
```
---
### **List: Epic B: Frontend**

**Card 1 of 6**
**Title:** `[B.1] Setup initial Streamlit dashboard UI`
**Description:**
```
**User Story:** As a trader (Alex), I want to see the primary signal, the alignment matrix, and the risk level on one screen so I can make a quick, informed decision.
---
**Acceptance Criteria:**
- A static Streamlit dashboard runs locally.
- It has clear, labeled placeholders for the Signal, Alignment Matrix, and Risk Meter.
---
**Estimate:** 1 day
**Owner:** Member 2
**Dependencies:** None
```

**Card 2 of 6**
**Title:** `[B.2] Build the PCS feedback component`
**Description:**
```
**User Story:** As a trader, I want to provide quick feedback on how the signal makes me feel so that I can reflect on my confidence.
---
**Acceptance Criteria:**
- A 1-5 star or button component is visible on the dashboard.
- Clicking it triggers a function and displays a "Thank You" message in the UI.
---
**Estimate:** 1 day
**Owner:** Member 2
**Dependencies:** Task B.1
```

**Card 3 of 6**
**Title:** `[B.3] Create API client service`
**Description:**
```
**User Story:** As a trader (Alex), I want to see the primary signal, the alignment matrix, and the risk level on one screen so I can make a quick, informed decision.
---
**Acceptance Criteria:**
- A Python function exists that can call the backend `/signal/btc` endpoint.
- It correctly parses the returned JSON and handles connection errors gracefully.
---
**Estimate:** 1 day
**Owner:** Member 2
**Dependencies:** Task A.6
```

**Card 4 of 6**
**Title:** `[B.4] Integrate API data into dashboard`
**Description:**
```
**User Story:** As a trader (Alex), I want to see the primary signal, the alignment matrix, and the risk level on one screen so I can make a quick, informed decision.
---
**Acceptance Criteria:**
- The dashboard calls the API client on page load.
- It correctly populates the UI placeholders with live data from the backend.
---
**Estimate:** 1 day
**Owner:** Member 2
**Dependencies:** Task B.3
```

**Card 5 of 6**
**Title:** `[B.5] Connect feedback component to backend`
**Description:**
```
**User Story:** As a trader, I want to provide quick feedback on how the signal makes me feel so that I can reflect on my confidence.
---
**Acceptance Criteria:**
- Clicking the feedback component in the UI now successfully sends the score to the `POST /feedback` endpoint on the backend.
---
**Estimate:** 1 day
**Owner:** Member 2
**Dependencies:** Task B.2, Task A.7
```

**Card 6 of 6**
**Title:** `[B.6] Refine UI & add data visualizations`
**Description:**
```
**User Story:** As a trader (Alex), I want to see the primary signal, the alignment matrix, and the risk level on one screen so I can make a quick, informed decision.
---
**Acceptance Criteria:**
- Use Plotly or another library to create more polished, professional-looking visualizations for the Alignment Matrix and Risk Meter.
- Improve the overall layout, fonts, and readability of the dashboard.
---
**Estimate:** 2 days
**Owner:** Member 2
**Dependencies:** Task B.4
```
---
### **List: Epic C: Deployment**

**Card 1 of 5**
**Title:** `[C.1] Dockerize the FastAPI backend`
**Description:**
```
**User Story:** As a developer, I want the application to be containerized so that it runs consistently everywhere.
---
**Acceptance Criteria:**
- A `Dockerfile` is created for the backend.
- `docker build` and `docker run` successfully starts the FastAPI server.
---
**Estimate:** 1 day
**Owner:** Member 1
**Dependencies:** Task A.6
```

**Card 2 of 5**
**Title:** `[C.2] Dockerize the Streamlit frontend`
**Description:**
```
**User Story:** As a developer, I want the application to be containerized so that it runs consistently everywhere.
---
**Acceptance Criteria:**
- A `Dockerfile` is created for the frontend.
- `docker build` and `docker run` successfully starts the Streamlit app.
---
**Estimate:** 1 day
**Owner:** Member 2
**Dependencies:** Task B.1
```

**Card 3 of 5**
**Title:** `[C.3] Configure Docker Compose for local dev`
**Description:**
```
**User Story:** As a developer, I want the application to be containerized so that it runs consistently everywhere.
---
**Acceptance Criteria:**
- A `docker-compose.yml` file is created.
- Running `docker-compose up` successfully starts both the backend and frontend services, and they can communicate with each other.
---
**Estimate:** 1 day
**Owner:** Both
**Dependencies:** Task C.1, Task C.2
```

**Card 4 of 5**
**Title:** `[C.4] Deploy application to a cloud service`
**Description:**
```
**User Story:** As a developer, I want the application to be deployed to the cloud so that our testers can access it.
---
**Acceptance Criteria:**
- The containerized application is running on a cloud host (e.g., Streamlit Cloud, Heroku, etc.).
- The application is accessible to the team via a public URL.
---
**Estimate:** 2 days
**Owner:** Both
**Dependencies:** Task C.3
```

**Card 5 of 5**
**Title:** `[C.5] Full end-to-end testing & bug bash`
**Description:**
```
**User Story:** As a developer, I want the application to be deployed to the cloud so that our testers can access it.
---
**Acceptance Criteria:**
- The team performs rigorous testing of all features on the live application.
- All critical bugs are logged in Trello and prioritized for fixing.
---
**Estimate:** 2 days
**Owner:** Both
**Dependencies:** Task C.4
```