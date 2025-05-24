import React, { useState, useEffect } from 'react';

// Mock ConfirmationPopup component for demonstration
const ConfirmationPopup = ({ onClose }) => (
  <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 animate-fadeIn">
    <div className="bg-white rounded-2xl p-8 max-w-md mx-4 animate-slideUp">
      <div className="text-center">
        <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4 animate-bounce">
          <span className="text-green-500 text-2xl">✓</span>
        </div>
        <h3 className="text-xl font-bold text-gray-800 mb-2">Review Submitted!</h3>
        <p className="text-gray-600 mb-6">Thank you for your feedback</p>
        <button
          onClick={onClose}
          className="bg-green-500 text-white px-6 py-2 rounded-full hover:bg-green-600 transition-all duration-300 transform hover:scale-105"
        >
          Done
        </button>
      </div>
    </div>
  </div>
);

const PositiveReviewPage = ({ restaurant = { name: "Sample Restaurant" }, onBack = () => {} }) => {
  const [showConfirm, setShowConfirm] = useState(false);
  const [foodOrdered, setFoodOrdered] = useState('');
  const [hygieneRating, setHygieneRating] = useState(0);
  const [likedMost, setLikedMost] = useState('');
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const handleSubmit = () => {
    const review = {
      restaurant: restaurant.name,
      foodOrdered,
      hygieneRating,
      likedMost
    };
    console.log('Positive Review Submitted:', review);
    setShowConfirm(true);
  };

  const renderStars = (rating, onClick) => (
    <div className="flex justify-center space-x-2 my-4">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          onClick={() => onClick(star)}
          className={`text-3xl transition-all duration-300 transform hover:scale-125 ${
            star <= rating 
              ? 'text-yellow-400 animate-pulse' 
              : 'text-gray-300 hover:text-yellow-200'
          }`}
        >
          ★
        </button>
      ))}
    </div>
  );

  return (
    <>
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        @keyframes slideUp {
          from { 
            opacity: 0;
            transform: translateY(30px);
          }
          to { 
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes slideInLeft {
          from { 
            opacity: 0;
            transform: translateX(-30px);
          }
          to { 
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
        
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(34, 197, 94, 0.3); }
          50% { box-shadow: 0 0 30px rgba(34, 197, 94, 0.6); }
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
        
        .animate-slideUp {
          animation: slideUp 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        }
        
        .animate-slideInLeft {
          animation: slideInLeft 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        }
        
        .animate-float {
          animation: float 3s ease-in-out infinite;
        }
        
        .animate-glow {
          animation: glow 2s ease-in-out infinite;
        }
        
        .input-focus:focus {
          transform: scale(1.02);
          box-shadow: 0 8px 25px rgba(34, 197, 94, 0.15);
        }
        
        .gradient-bg {
          background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        }
        
        .card-shadow {
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        }
        
        .hover-lift:hover {
          transform: translateY(-2px);
          box-shadow: 0 25px 70px rgba(0, 0, 0, 0.15);
        }

        /* Hide scrollbar */
        ::-webkit-scrollbar {
          display: none; /* Safari and Chrome */
        }
        body {
          scrollbar-width: none; /* Firefox */
        }
      `}</style>

      <div className="min-h-screen gradient-bg p-6 flex items-center justify-center overflow-hidden">
        {showConfirm && <ConfirmationPopup onClose={onBack} />}
        
        <div className={`w-full max-w-2xl bg-white rounded-3xl card-shadow p-8 transition-all duration-700 ${
          isVisible ? 'animate-slideUp opacity-100' : 'opacity-0'
        }`}>
          
          {/* Header with floating icon */}
          <div className="text-center mb-8">
            <div className="inline-block animate-float mb-4">
              <div className="w-20 h-20 bg-gradient-to-br from-green-400 to-green-600 rounded-full flex items-center justify-center animate-glow">
                <span className="text-white text-3xl">😊</span>
              </div>
            </div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-green-600 to-green-800 bg-clip-text text-transparent mb-2 animate-slideInLeft">
              You liked {restaurant.name}!
            </h2>
            <div className="w-24 h-1 bg-gradient-to-r from-green-400 to-green-600 rounded-full mx-auto"></div>
          </div>

          {/* Form Fields */}
          <div className="space-y-8">
            {/* Food Ordered */}
            <div className="animate-slideInLeft" style={{ animationDelay: '0.1s' }}>
              <label className="block text-lg font-semibold text-gray-700 mb-3">
                What food did you order? 🍽️
              </label>
              <input
                type="text"
                value={foodOrdered}
                onChange={(e) => setFoodOrdered(e.target.value)}
                className="w-full border-2 border-gray-200 p-4 rounded-2xl input-focus transition-all duration-300 text-gray-800 placeholder-gray-400 hover-lift"
                placeholder="e.g. Paneer Tikka, Pasta, Pizza..."
              />
            </div>

            {/* Hygiene Rating */}
            <div className="animate-slideInLeft" style={{ animationDelay: '0.2s' }}>
              <label className="block text-lg font-semibold text-gray-700 mb-3 text-center">
                Rate hygiene 🧼
              </label>
              <div className="bg-gray-50 rounded-2xl p-6 hover-lift transition-all duration-300">
                {renderStars(hygieneRating, setHygieneRating)}
                {hygieneRating > 0 && (
                  <p className="text-center text-green-600 font-medium mt-2 animate-fadeIn">
                    {hygieneRating === 5 ? 'Excellent!' : 
                     hygieneRating === 4 ? 'Very Good!' : 
                     hygieneRating === 3 ? 'Good!' : 
                     hygieneRating === 2 ? 'Fair' : 'Needs Improvement'}
                  </p>
                )}
              </div>
            </div>

            {/* What Liked Most */}
            <div className="animate-slideInLeft" style={{ animationDelay: '0.3s' }}>
              <label className="block text-lg font-semibold text-gray-700 mb-3">
                What did you like the most? ❤️
              </label>
              <input
                type="text"
                value={likedMost}
                onChange={(e) => setLikedMost(e.target.value)}
                className="w-full border-2 border-gray-200 p-4 rounded-2xl input-focus transition-all duration-300 text-gray-800 placeholder-gray-400 hover-lift"
                placeholder="e.g. Amazing taste, Friendly service, Great ambience..."
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-between animate-slideInLeft" style={{ animationDelay: '0.4s' }}>
            <button 
              onClick={onBack} 
              className="group bg-gray-100 hover:bg-gray-200 text-gray-700 px-8 py-4 rounded-2xl transition-all duration-300 transform hover:scale-105 flex items-center justify-center font-medium"
            >
              <span className="mr-2 transition-transform duration-300 group-hover:-translate-x-1">←</span>
              Back
            </button>
            
            <button
              onClick={handleSubmit}
              disabled={!foodOrdered || !hygieneRating || !likedMost}
              className={`px-8 py-4 rounded-2xl font-medium transition-all duration-300 transform flex items-center justify-center ${
                foodOrdered && hygieneRating && likedMost
                  ? 'bg-gradient-to-r from-green-500 to-green-600 text-white hover:from-green-600 hover:to-green-700 hover:scale-105 animate-glow'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              Submit Review
              <span className="ml-2 transition-transform duration-300 hover:translate-x-1">✨</span>
            </button>
          </div>

          {/* Progress indicator */}
          <div className="mt-8 flex justify-center space-x-2">
            {[foodOrdered, hygieneRating, likedMost].map((field, index) => (
              <div
                key={index}
                className={`w-3 h-3 rounded-full transition-all duration-300 ${
                  field ? 'bg-green-500' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>
        </div>
      </div>
    </>
  );
};

export default PositiveReviewPage;
