import React, { useEffect, useRef } from 'react';

/**
 * Floating popup component positioned absolutely
 * Shows main content and optional side info panel
 * Closes on clicking outside
 * 
 * Props:
 * - isOpen: boolean, whether popup is shown
 * - onClose: function to close popup
 * - anchorPosition: {x, y} coordinates for positioning popup
 * - children: main content (e.g., restaurant name, location)
 * - sideContent: optional side panel content (e.g., details)
 */
const Popup = ({ isOpen, onClose, anchorPosition = { x: 0, y: 0 }, children, sideContent }) => {
  const popupRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (event) => {
      if (popupRef.current && !popupRef.current.contains(event.target)) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      ref={popupRef}
      style={{
        position: 'absolute',
        top: anchorPosition.y,
        left: anchorPosition.x,
        zIndex: 1000,
        display: 'flex',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        borderRadius: '12px',
        background: 'white',
        overflow: 'hidden',
        animation: 'fadeIn 0.2s ease-out forwards',
        minWidth: 280,
        maxWidth: 400,
      }}
    >
      {/* Main popup content */}
      <div style={{ padding: 16, flex: 1 }}>
        {children}
      </div>

      {/* Optional right side info panel */}
      {sideContent && (
        <div
          style={{
            width: 180,
            borderLeft: '1px solid #ddd',
            backgroundColor: '#f9f9f9',
            padding: 16,
            overflowY: 'auto',
          }}
        >
          {sideContent}
        </div>
      )}

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default Popup;
