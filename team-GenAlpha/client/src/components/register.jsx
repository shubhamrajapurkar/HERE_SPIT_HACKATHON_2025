import { createUserWithEmailAndPassword } from "firebase/auth";
import React, { useState } from "react";
import { auth, db } from "./firebase";
import { setDoc, doc } from "firebase/firestore";
import { toast } from "react-toastify";
import { useNavigate } from "react-router-dom";

function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      if (user) {
        // Store user data in Users collection
        const userData = {
          email: user.email,
          name: name,
          createdAt: new Date(),
          playlist: [], // Initialize empty playlist array
        };

        // Store user data in Firestore
        await setDoc(doc(db, "Users", user.uid), userData);
        
        toast.success("User Registered Successfully!", {
          position: "top-center",
        });
        navigate("/dashboard");
      }
    } catch (error) {
      console.error(error.message);
      toast.error(error.message, {
        position: "bottom-center",
      });
    }
  };

  return (
    <form onSubmit={handleRegister} className="flex flex-col items-center p-6 max-w-md mx-auto bg-white shadow-md rounded-lg">
      <h3 className="text-xl font-semibold mb-4">Sign Up</h3>

      <div className="mb-4 w-full">
        <label className="block text-sm font-medium mb-1">Name</label>
        <input
          type="text"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:ring-blue-500"
          placeholder="Enter name"
          onChange={(e) => setName(e.target.value)}
          required
        />
      </div>

      <div className="mb-4 w-full">
        <label className="block text-sm font-medium mb-1">Email address</label>
        <input
          type="email"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:ring-blue-500"
          placeholder="Enter email"
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </div>

      <div className="mb-4 w-full">
        <label className="block text-sm font-medium mb-1">Password</label>
        <input
          type="password"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:ring-blue-500"
          placeholder="Enter password"
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </div>

      <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700">
        Sign Up
      </button>

      <p className="text-sm mt-4">
        Already registered? <a href="/" className="text-blue-600 hover:underline">Login</a>
      </p>
    </form>
  );
}

export default Register;