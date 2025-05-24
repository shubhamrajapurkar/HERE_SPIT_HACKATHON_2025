import React, { useState, useEffect } from 'react';
import Modal from './Modal';
import restaurantData from '../../data/restaurant_review.json';
import PositiveReviewPage from './PositiveReviewPage';
import NegativeReviewPage from './NegativeReviewPage';

const UserReviewsPage = ({ isOpen, onClose }) => {
  const [query, setQuery] = useState('');
  const [filtered, setFiltered] = useState([]);
  const [selected, setSelected] = useState(null);
  const [reviewType, setReviewType] = useState(null); // 'good' | 'bad' | null

  useEffect(() => {
    if (query.length > 0) {
      const results = restaurantData.filter(r =>
        r.name.toLowerCase().includes(query.toLowerCase())
      );
      setFiltered(results);
    } else {
      setFiltered([]);
    }
  }, [query]);

  const handleBack = () => {
    setReviewType(null);
  };

  const handleSelectRestaurant = (item) => {
    setSelected(item);
    setFiltered([]);
    setQuery(item.name);
  };

  const handleSelectReview = (type) => {
    if (!selected) return; // Prevent if no restaurant selected
    setReviewType(type);
  };

  useEffect(() => {
    if (!isOpen) {
      setQuery('');
      setFiltered([]);
      setSelected(null);
      setReviewType(null);
    }
  }, [isOpen]);

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="p-6 min-h-[300px] transition-all duration-300 relative">
        {!reviewType && (
          <>
            <h2 className="text-2xl font-bold mb-4">Rate a Restaurant</h2>

            <div className="flex items-center space-x-2 mb-2 max-w-xl">
              <input
                type="text"
                placeholder="Search for a restaurant..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="flex-grow p-3 border rounded-lg transition duration-300 focus:outline-none focus:ring-2 focus:ring-green-400"
              />
              <button
                aria-label="Dislike"
                onClick={() => handleSelectReview('bad')}
                disabled={!selected}
                className={
                  `w-12 h-12 flex items-center justify-center rounded-full bg-red-300 hover:bg-red-400 transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed` +
                  (!selected ? ' pointer-events-none' : '')
                }
                title="Dislike (Go to negative review)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-white transform rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" >
                  <path d="M9 5l7 7-7 7" />
                </svg>
              </button>
              <button
                aria-label="Like"
                onClick={() => handleSelectReview('good')}
                disabled={!selected}
                className={
                  `w-12 h-12 flex items-center justify-center rounded-full bg-green-400 hover:bg-green-500 transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed` +
                  (!selected ? ' pointer-events-none' : '')
                }
                title="Like (Go to positive review)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" >
                  <path d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>

            {filtered.length > 0 && (
              <ul className="border rounded-lg mb-4 max-h-40 overflow-y-auto transition duration-300 max-w-xl">
                {filtered.map((item) => (
                  <li
                    key={item.id}
                    onClick={() => handleSelectRestaurant(item)}
                    className={
                      `p-2 cursor-pointer hover:bg-gray-100 transition duration-300 ` +
                      (selected?.id === item.id ? 'bg-green-100 font-semibold' : '')
                    }
                  >
                    {item.name}
                  </li>
                ))}
              </ul>
            )}

            {selected && (
              <div className="p-6 border rounded-xl shadow-md text-center transition duration-300 transform hover:scale-105 max-w-xl">
                <h3 className="text-xl font-semibold mb-2">{selected.name}</h3>
                <p>Click the green arrow to like, or the red arrow to dislike</p>
              </div>
            )}
          </>
        )}

        {reviewType === 'good' && selected && (
          <div className="transition-transform duration-500 translate-x-0">
            <PositiveReviewPage restaurant={selected} onBack={handleBack} />
          </div>
        )}
        {reviewType === 'bad' && selected && (
          <div className="transition-transform duration-500 translate-x-0">
            <NegativeReviewPage restaurant={selected} onBack={handleBack} />
          </div>
        )}
      </div>
    </Modal>
  );
};

export default UserReviewsPage;
