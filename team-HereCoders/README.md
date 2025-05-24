# Team HereCoders

Welcome to **Sprint Sync**, an Expo-based React Native project leveraging the power of Expo Router, Firebase, and various other libraries to build a modern cross-platform app for iOS, Android, and Web.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dependencies & Libraries](#dependencies--libraries)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Resetting the Project](#resetting-the-project)
- [Learn More](#learn-more)
- [Community](#community)

---

## Project Overview

This project is bootstrapped with [`create-expo-app`](https://www.npmjs.com/package/create-expo-app) and uses [Expo Router](https://expo.github.io/router/docs) for file-based routing. It integrates Firebase for authentication and backend services and MongoDB/Mongoose for database management on the backend server.

The app is designed to run seamlessly across Android, iOS, and web platforms.

---

## Features

- File-based routing with Expo Router
- Firebase Authentication and Realtime Database integration
- Async Storage for local data persistence
- Secure storage with Expo Secure Store
- Location and image picker support using Expo APIs
- Web, iOS, and Android support with one codebase
- Backend powered by Express with MongoDB and Mongoose
- JWT based authentication handling
- UI enhanced with vector icons and blur effects
- Robust state and network request handling with axios

---

## Dependencies & Libraries

### Core Expo and React Native

- `expo` - Expo SDK
- `react-native` - React Native core framework
- `expo-router` - Routing for Expo apps
- `react-navigation` & related libraries - Navigation support
- `expo-image-picker`, `expo-location`, `expo-secure-store`, `expo-blur` - Expo APIs for native features
- `@expo/vector-icons` - Icons pack
- `react-native-gesture-handler`, `react-native-reanimated`, `react-native-safe-area-context`, `react-native-screens` - Gesture and UI helpers

### Backend and Authentication

- `firebase` - Firebase client SDK
- `@react-native-firebase/app` & `@react-native-firebase/auth` - Firebase native modules for React Native
- `express` - Node.js backend framework
- `mongodb` & `mongoose` - MongoDB ODM for database operations
- `jsonwebtoken` - JWT handling for secure authentication
- `bcrypt` - Password hashing

### Utilities

- `axios` - Promise-based HTTP client
- `body-parser`, `cors` - Express middleware for requests

### Development

- `typescript` - TypeScript support
- `eslint` & `eslint-config-expo` - Linting and code quality tools
- Babel core for JS compilation

---

## Getting Started

### Prerequisites

- Node.js (recommended version >= 18)
- npm or yarn
- Expo CLI (`npm install -g expo-cli` recommended)
- Android Studio or Xcode for native simulators/emulators (optional)

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/reprojhackathon.git
cd reprojhackathon
```

Install dependencies:

```bash
npm install
# or
yarn install
```

---

## Running the Project

Start the development server:

```bash
npm start
# or
expo start
```

This will open the Expo developer tools in your browser where you can:

- Run the app on an Android emulator or device
- Run the app on an iOS simulator or device
- Open the app in Expo Go (mobile app)
- Run the app on web

You can also use the scripts:

- `npm run android` — open Android emulator/device
- `npm run ios` — open iOS simulator/device
- `npm run web` — open the project in a web browser

---

## Resetting the Project

If you want to reset the project to a fresh state (removing all your current app code and starting from scratch):

```bash
npm run reset-project
```

This moves your current app code to an `app-example` directory and creates a new blank `app` directory to start fresh.

---

## Learn More

- [Expo Documentation](https://docs.expo.dev/)
- [Expo Router Documentation](https://expo.github.io/router/docs)
- [React Navigation](https://reactnavigation.org/)
- [Firebase Documentation](https://firebase.google.com/docs)
- [MongoDB and Mongoose](https://mongoosejs.com/docs/)

---

## Community

Join the Expo and React Native communities:

- [Expo GitHub](https://github.com/expo/expo)
- [Expo Discord Chat](https://chat.expo.dev)
- [React Native Community](https://reactnative.dev/community)

---

If you have any questions or want to contribute, feel free to open an issue or a pull request.

---

_Happy coding!_ 🚀

---
