import React from 'react';

const Modal = ({ isOpen, onClose, children }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <div className="relative">
          <button 
            onClick={onClose}
            className="absolute right-4 top-4 text-gray-500 hover:text-gray-700 z-50"
          >
            ✕
          </button>
          {children}
        </div>
      </div>
    </div>
  );
};

export default Modal;