### **CryptoPulse AI - MVP Sprint Kanban Board**

Here’s a look at our 3-week plan, showing which tasks we'll tackle each week. We can move cards to an "In Progress" or "Done" column in Trello as we work on them.

```
========================================================================================================================
| BACKLOG (Epics)                             | WEEK 1 (Setup & Foundations)                | WEEK 2 (Integration & Core Logic)           | WEEK 3 (Deployment & Refinement)            |
========================================================================================================================
|                                             |                                             |                                             |                                             |
| 🧱 EPIC A: Core Signal Engine (Backend)     | [A.1] Setup data ingestion script (M1)      | [A.3] Train baseline classification model (M1)| [A.7] Add basic analytics logging (M1)      |
|                                             |                                             |                                             |                                             |
| 🧱 EPIC B: Signal Dashboard (Frontend)      | [A.2] Engineer features for model (M1)      | [A.5] Build signal generation logic (M1)      | [B.5] Connect feedback component to backend (M2)|
|                                             |                                             |                                             |                                             |
| 🧱 EPIC C: Deployment & Testing             | [A.4] Create FastAPI app structure (M1)     | [A.6] Implement /signal/btc endpoint (M1)     | [B.6] Refine UI & add data visualizations (M2)|
|                                             |                                             |                                             |                                             |
|                                             | [B.1] Setup initial Streamlit dashboard (M2)| [B.3] Create API client service (M2)        | [C.4] Deploy application to cloud (Both)    |
|                                             |                                             |                                             |                                             |
|                                             | [C.1] Dockerize the FastAPI backend (M1)     | [B.4] Integrate API data into dashboard (M2)  | [C.5] Full end-to-end testing & bug bash (Both)|
|                                             |                                             |                                             |                                             |
|                                             | [C.2] Dockerize the Streamlit frontend (M2) | [C.3] Configure Docker Compose (Both)       |                                             |
|                                             |                                             |                                             |                                             |
------------------------------------------------------------------------------------------------------------------------
```

### **How to Read This Board:**

*   **BACKLOG:** These are our three main themes of work (the Epics).
*   **WEEK 1:** The focus is on setting up the foundations. By the end of this week, we aim to have a basic, non-connected frontend and backend running inside Docker.
*   **WEEK 2:** This is integration week. The critical goal is to get the frontend talking to the backend (`[A.6]` -> `[B.3]` -> `[B.4]`). We'll also build out the core model and signal logic.
*   **WEEK 3:** The final push is about deploying the application to the cloud, making it accessible for testing, and refining the user experience based on our own feedback before showing it to testers.