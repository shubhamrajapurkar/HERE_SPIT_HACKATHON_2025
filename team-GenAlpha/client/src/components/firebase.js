import { initializeApp } from "firebase/app";
import {getAuth} from "firebase/auth";
import {getFirestore} from "firebase/firestore"
import { getStorage } from "firebase/storage";
const firebaseConfig = {
  apiKey: "AIzaSyB5YyEn_Tk3IEF7CKIAlGLz7QuvZm6149I",
  authDomain: "tsec-hacks-3c46c.firebaseapp.com",
  projectId: "tsec-hacks-3c46c",
  storageBucket: "tsec-hacks-3c46c.firebasestorage.app",
  messagingSenderId: "690028308588",
  appId: "1:690028308588:web:ef37a525758aafe128d049"
};

const app = initializeApp(firebaseConfig);
export const storage = getStorage(app);


export const auth=getAuth(app);

export const db = getFirestore(app);
export default app;