import React from 'react';

const ConfirmationPopup = ({ onClose }) => (
  <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
    <div className="bg-white rounded-2xl p-6 shadow-xl text-center max-w-sm w-full">
      <h2 className="text-xl font-bold mb-3">Thank you for your feedback!</h2>
      <p className="text-gray-600 mb-4">Your review has been recorded successfully.</p>
      <button
        onClick={onClose}
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Close
      </button>
    </div>
  </div>
);

export default ConfirmationPopup;
