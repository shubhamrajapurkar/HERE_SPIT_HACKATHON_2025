
---

# 🚨 Emergencio App - Emergency Response System

A comprehensive emergency response mobile application built for the **HERE SPIT Hackathon 2025** by **Team ZenithX**. This app helps users quickly locate and navigate to emergency services during critical situations using HERE Maps and Twilio for real-time communication.

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🛠️ Tech Stack](#-tech-stack)
- [📋 Prerequisites](#-prerequisites)
- [📦 Installation](#-installation)
- [🔑 API Keys Setup](#-api-keys-setup)
- [🚀 Running the Application](#-running-the-application)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [👨‍💻 Team](#-team)

---

## ✨ Features

- 🗺️ Real-time Maps Integration with HERE Technologies
- 📍 GPS Location Tracking
- 🚨 Emergency Services Locator (Hospitals, Police, Fire Stations)
- 🧭 Turn-by-turn Navigation with Alternate Routes and Live Traffic
- 📞 One-Tap SOS Messaging and Calling (Twilio)
- ⚠️ Geofencing Alerts
- 📱 Cross-platform Mobile App (iOS & Android)
- 🔄 Fast and reliable backend API support

---

## 🛠️ Tech Stack

### 🖥️ Frontend (React Native + Expo)
- React Native `0.79.2`
- Expo SDK `~53.0.9`
- React Navigation `7.x`
- HERE Maps SDK
- `expo-location`

### 🔧 Backend (Node.js + Express)
- Express framework
- HERE Routing & Search APIs
- Twilio API for SMS & Voice
- RESTful API Architecture

---

## 📋 Prerequisites

Make sure you have the following installed:

- [Node.js (v16+)](https://nodejs.org/)
- npm (comes with Node.js)
- [Expo CLI](https://docs.expo.dev/workflow/expo-cli/)  
  Install it globally if not already:
  ```bash
  npm install -g expo-cli

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/emergencio-app.git
cd emergencio-app
```

### 2. Install dependencies

#### Frontend

```bash
cd frontend
npm install
```

#### Backend

```bash
cd ../backend
npm install
```

---

## 🔑 API Keys Setup

You need two external API keys to run this project:

1. **HERE API Key** – For maps, routing, geocoding, and POI search
   👉 [Get it from HERE Developer Portal](https://developer.here.com/)

2. **Twilio Credentials** – For sending SMS and making emergency calls
   👉 [Get it from Twilio](https://www.twilio.com/try-twilio)

### Backend `.env` Configuration

Create a `.env` file inside the `backend/` directory and add the following:

```env
HERE_API_KEY=your_here_api_key
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
```

---

## 🚀 Running the Application

### Start Backend (Node.js Server)

```bash
cd backend
npm run start
```

The backend server will start on `http://localhost:5000`.

### Start Frontend (Expo React Native App)

```bash
cd ../frontend
npx expo start
```

Use the QR code to run the app on your phone using the Expo Go app, or choose to open it in an iOS/Android simulator.

---

## 📁 Project Structure

```
emergencio-app/
├── backend/              # Node.js backend
│   ├── routes/
│   ├── controllers/
│   ├── .env
│   └── server.js
├── frontend/             # React Native frontend (Expo)
│   ├── components/
│   ├── screens/
│   └── App.js
└── README.md
```

---

## 🤝 Contributing

We welcome contributions from developers and enthusiasts!

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Make your changes
4. Commit (`git commit -m "Add feature"`)
5. Push to GitHub (`git push origin feature-name`)
6. Open a Pull Request 🎉

---

## 👨‍💻 Team – ZenithX

Built at  HERE SPIT Hackathon 2025

---
