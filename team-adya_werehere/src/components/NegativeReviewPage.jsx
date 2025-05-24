import React, { useState, useEffect } from 'react';

// Mock ConfirmationPopup component for demonstration
const ConfirmationPopup = ({ onClose }) => (
  <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 animate-fadeIn">
    <div className="bg-white rounded-2xl p-8 max-w-md mx-4 animate-slideUp">
      <div className="text-center">
        <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4 animate-bounce">
          <span className="text-orange-500 text-2xl">📝</span>
        </div>
        <h3 className="text-xl font-bold text-gray-800 mb-2">Feedback Submitted</h3>
        <p className="text-gray-600 mb-6">Thank you for helping us improve</p>
        <button
          onClick={onClose}
          className="bg-orange-500 text-white px-6 py-2 rounded-full hover:bg-orange-600 transition-all duration-300 transform hover:scale-105"
        >
          Done
        </button>
      </div>
    </div>
  </div>
);

const NegativeReviewPage = ({ restaurant = { name: "Sample Restaurant" }, onBack = () => {} }) => {
  const [showConfirm, setShowConfirm] = useState(false);
  const [issuesFaced, setIssuesFaced] = useState('');
  const [hygieneRating, setHygieneRating] = useState(0);
  const [comments, setComments] = useState('');
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const handleSubmit = () => {
    const review = {
      restaurant: restaurant.name,
      issuesFaced,
      hygieneRating,
      comments
    };
    console.log('Negative Review Submitted:', review);
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
              ? 'text-red-400 animate-pulse' 
              : 'text-gray-300 hover:text-red-200'
          }`}
        >
          ★
        </button>
      ))}
    </div>
  );

  const issueOptions = [
    { value: '', label: 'Select issue', icon: '❓' },
    { value: 'poor service', label: 'Poor Service', icon: '🙄' },
    { value: 'bad taste', label: 'Bad Taste', icon: '😷' },
    { value: 'dirty environment', label: 'Dirty Environment', icon: '🧹' },
    { value: 'long wait time', label: 'Long Wait Time', icon: '⏰' },
    { value: 'rude staff', label: 'Rude Staff', icon: '😠' },
    { value: 'other', label: 'Other', icon: '💭' }
  ];

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
        
        @keyframes gentleFloat {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-5px) rotate(2deg); }
        }
        
        @keyframes softGlow {
          0%, 100% { box-shadow: 0 0 20px rgba(251, 146, 60, 0.2); }
          50% { box-shadow: 0 0 30px rgba(251, 146, 60, 0.4); }
        }
        
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-2px); }
          75% { transform: translateX(2px); }
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
        
        .animate-gentleFloat {
          animation: gentleFloat 4s ease-in-out infinite;
        }
        
        .animate-softGlow {
          animation: softGlow 3s ease-in-out infinite;
        }
        
        .animate-shake {
          animation: shake 0.5s ease-in-out;
        }
        
        .input-focus:focus {
          transform: scale(1.02);
          box-shadow: 0 8px 25px rgba(251, 146, 60, 0.15);
        }
        
        .gradient-bg {
          background: linear-gradient(135deg, #fef7ed 0%, #fed7aa 50%, #fdba74 100%);
        }
        
        .card-shadow {
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        }
        
        .hover-lift:hover {
          transform: translateY(-2px);
          box-shadow: 0 25px 70px rgba(0, 0, 0, 0.15);
        }
        
        .custom-select {
          background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3e%3c/svg%3e");
          background-position: right 0.5rem center;
          background-repeat: no-repeat;
          background-size: 1.5em 1.5em;
          padding-right: 2.5rem;
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
          
          {/* Header with empathetic design */}
          <div className="text-center mb-8">
            <div className="inline-block animate-gentleFloat mb-4">
              <div className="w-20 h-20 bg-gradient-to-br from-orange-400 to-red-500 rounded-full flex items-center justify-center animate-softGlow">
                <span className="text-white text-3xl">😔</span>
              </div>
            </div>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent mb-2 animate-slideInLeft leading-relaxed">
              Sorry you had a bad experience at {restaurant.name}
            </h2>
            <p className="text-gray-600 animate-slideInLeft" style={{ animationDelay: '0.1s' }}>
              Your feedback helps us improve for everyone
            </p>
            <div className="w-24 h-1 bg-gradient-to-r from-orange-400 to-red-500 rounded-full mx-auto mt-3"></div>
          </div>

          {/* Form Fields */}
          <div className="space-y-8">
            {/* Issue Selection */}
            <div className="animate-slideInLeft" style={{ animationDelay: '0.2s' }}>
              <label className="block text-lg font-semibold text-gray-700 mb-3">
                What issues did you face? 🤔
              </label>
              <div className="relative">
                <select
                  value={issuesFaced}
                  onChange={(e) => setIssuesFaced(e.target.value)}
                  className="w-full border-2 border-gray-200 p-4 rounded-2xl input-focus transition-all duration-300 text-gray-800 hover-lift custom-select appearance-none bg-white"
                >
                  {issueOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.icon} {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Hygiene Rating */}
            <div className="animate-slideInLeft" style={{ animationDelay: '0.3s' }}>
              <label className="block text-lg font-semibold text-gray-700 mb-3 text-center">
                Rate hygiene 🧼
              </label>
              <div className="bg-red-50 rounded-2xl p-6 hover-lift transition-all duration-300">
                {renderStars(hygieneRating, setHygieneRating)}
                {hygieneRating > 0 && (
                  <p className="text-center text-red-600 font-medium mt-2 animate-fadeIn">
                    {hygieneRating === 1 ? 'Very Poor' : 
                     hygieneRating === 2 ? 'Poor' : 
                     hygieneRating === 3 ? 'Average' : 
                     hygieneRating === 4 ? 'Good' : 'Excellent'}
                  </p>
                )}
              </div>
            </div>

            {/* Additional Comments */}
            <div className="animate-slideInLeft" style={{ animationDelay: '0.4s' }}>
              <label className="block text-lg font-semibold text-gray-700 mb-3">
                Any additional comments? 💬
              </label>
              <textarea
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                className="w-full border-2 border-gray-200 p-4 rounded-2xl input-focus transition-all duration-300 text-gray-800 placeholder-gray-400 hover-lift resize-none"
                rows={4}
                placeholder="Tell us more about what went wrong... We genuinely want to improve."
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-between animate-slideInLeft" style={{ animationDelay: '0.5s' }}>
            <button 
              onClick={onBack} 
              className="group bg-gray-100 hover:bg-gray-200 text-gray-700 px-8 py-4 rounded-2xl transition-all duration-300 transform hover:scale-105 flex items-center justify-center font-medium"
            >
              <span className="mr-2 transition-transform duration-300 group-hover:-translate-x-1">←</span>
              Back
            </button>
            
            <button
              onClick={handleSubmit}
              disabled={!issuesFaced || !hygieneRating}
              className={`px-8 py-4 rounded-2xl font-medium transition-all duration-300 transform flex items-center justify-center ${
                issuesFaced && hygieneRating
                  ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white hover:from-orange-600 hover:to-red-600 hover:scale-105 animate-softGlow'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              Submit Feedback
              <span className="ml-2 transition-transform duration-300 hover:translate-x-1">📝</span>
            </button>
          </div>

          {/* Progress indicator */}
          <div className="mt-8 flex justify-center space-x-2">
            {[issuesFaced, hygieneRating, comments].map((field, index) => (
              <div
                key={index}
                className={`w-3 h-3 rounded-full transition-all duration-300 ${
                  field ? 'bg-orange-500' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>

          {/* Encouraging message */}
          <div className="mt-6 text-center text-sm text-gray-500 animate-fadeIn" style={{ animationDelay: '0.6s' }}>
            <p>💝 Your honest feedback helps create better experiences for everyone</p>
          </div>
        </div>
      </div>
    </>
  );
};

export default NegativeReviewPage;
