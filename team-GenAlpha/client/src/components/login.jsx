// Login.js
import { signInWithEmailAndPassword } from "firebase/auth";
import React, { useState } from "react";
import { auth } from "./firebase";
import { toast } from "react-toastify";
import { useNavigate } from "react-router-dom";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      await signInWithEmailAndPassword(auth, email, password);
      toast.success("User logged in Successfully!", {
        position: "top-center",
      });
      navigate("/dashboard");
    } catch (error) {
      toast.error(error.message, {
        position: "bottom-center",
      });
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col items-center p-6 max-w-md mx-auto bg-white shadow-md rounded-lg"
    >
      <h3 className="text-xl font-semibold mb-4">Login</h3>

      <div className="mb-4 w-full">
        <label className="block text-sm font-medium mb-1">Email address</label>
        <input
          type="email"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:ring-blue-500"
          placeholder="Enter email"
          value={email}
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
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </div>
      
      <button
        type="submit"
        className="w-full bg-blue-600 text-white py-2 px-2 rounded-md hover:bg-blue-700"
      >
        Login
      </button>

      <p className="text-sm mt-4">
        New user?{" "}
        <a href="/register" className="text-blue-600 hover:underline">
          Register Here
        </a>
      </p>
    </form>
  );
}

export default Login;