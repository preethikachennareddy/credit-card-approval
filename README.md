<div align="center">

# PracticeLeet

**AI-powered mock technical interview coach for Amazon, Google, Meta, Microsoft, and Apple**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.0--flash--lite-4285F4?logo=google&logoColor=white)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![google-generativeai](https://img.shields.io/badge/google--generativeai-0.8-4285F4?logo=google&logoColor=white)](https://pypi.org/project/google-generativeai)
[![CSS Modules](https://img.shields.io/badge/CSS-Modules-264de4?logo=css3&logoColor=white)](https://github.com/css-modules/css-modules)

### [View Live Demo](https://practiceleet-preethika.streamlit.app)

</div>

---

## Features

- **AI Interviewer** - In-character for Amazon, Google, Meta, Microsoft, or Apple. Gives one real coding problem, responds to clarifying questions, pushes for complexity analysis, gives hints only when requested (and tracks them).
- **Code Editor** - Write in Python, Java, C++, or JavaScript. Run visible test cases instantly.
- **LLM Feedback** - Correctness, complexity analysis, code quality, communication score, edge cases, hire/no-hire decision.
- **Progress Tracker** - Saves all attempts to localStorage. Tracks weak topics, scores over time, and streaks.
- **Company Modes** - Amazon (LP + coding), Google (communication + optimal), Meta (speed + patterns).

---

## Tech Stack

- **Frontend**: React 18
- **AI**: Google Gemini API (gemini-2.0-flash-lite)
- **Storage**: localStorage (upgrade to Supabase/PostgreSQL for multi-device)
- **Styling**: Pure CSS (no framework needed)

---

## Run locally (React version)

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/practiceleet.git
cd practiceleet
```

### 2. Install dependencies

```bash
npm install
```

### 3. Add your Gemini API key

```bash
cp .env.example .env
# Edit .env and add your key from https://aistudio.google.com/app/apikey
```

### 4. Run

```bash
npm start
```

Open [http://localhost:3000](http://localhost:3000).

---

## Project Structure

```
src/
  components/
    SetupScreen.jsx       - Company / topic / difficulty picker
    InterviewScreen.jsx   - Split-panel interview room
    ChatPanel.jsx         - AI interviewer chat
    CodePanel.jsx         - Code editor + test runner
    FeedbackScreen.jsx    - Post-interview scorecard
    ProgressDashboard.jsx - Historical attempts + weak topics
  hooks/
    useTimer.js           - Interview timer
    useProgress.js        - localStorage read/write
    useClaude.js          - Gemini API calls
  utils/
    prompts.js            - All system prompts
    companyModes.js       - Company-specific interview styles
    storage.js            - localStorage helpers
  pages/
    App.jsx               - Root router / screen switcher
  index.js
  index.css
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key (get free at aistudio.google.com/app/apikey) |

> **Security note**: This app calls the Gemini API directly from the browser. For production, proxy API calls through your own backend so the key is never exposed to clients.
