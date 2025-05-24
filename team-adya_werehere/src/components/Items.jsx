import { useState } from "react";
import { PiForkKnifeFill } from "react-icons/pi";
import { useNavigate } from "react-router-dom";
import {
  MdHotel,
  MdOutlineMuseum,
  MdSettings,
  MdTheaters,
} from "react-icons/md";
import { FaRoute } from "react-icons/fa";
import { MdRestaurant } from "react-icons/md";
import "./Items.css";
import Modal from "./Modal";
import UserPreferencePage from "../Pages/userPreferance";
import UserReviewsPage from "./UserReviewsPage";
import EntertainmentPage from "../Pages/EntertainmentPage";

const Items = () => {
  const [isPreferencesOpen, setIsPreferencesOpen] = useState(false);
  const [isReviewsOpen, setIsReviewsOpen] = useState(false);
  const [isEntertainmentOpen, setIsEntertainmentOpen] = useState(false);
  const navigate = useNavigate();

  return (
    <>
      <style jsx="true">{`
        .sidebar {
          animation: slideInLeft 0.6s ease-out;
        }

        .sidebar-icon {
          position: relative;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          overflow: hidden;
        }

        .sidebar-icon::before {
          content: "";
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          border-radius: inherit;
          opacity: 0;
          transition: all 0.3s ease;
          z-index: -1;
        }

        .sidebar-icon:nth-child(1)::before {
          background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        }

        .sidebar-icon:nth-child(2)::before {
          background: linear-gradient(135deg, #4ecdc4, #44a08d);
        }

        .sidebar-icon:nth-child(3)::before {
          background: linear-gradient(135deg, #a8a8a8, #7d7d7d);
        }

        .sidebar-icon:nth-child(4)::before {
          background: linear-gradient(135deg, #45b7d1, #96c93d);
        }

        .sidebar-icon:nth-child(5)::before {
          background: linear-gradient(135deg, #f093fb, #f5576c);
        }

        .sidebar-icon:hover {
          transform: translateY(-3px) scale(1.05);
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        .sidebar-icon:hover::before {
          opacity: 1;
        }

        .sidebar-icon:active {
          transform: translateY(-1px) scale(0.98);
        }

        .sidebar-icon svg {
          transition: all 0.3s ease;
          filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
        }

        .sidebar-icon:hover svg {
          color: white !important;
          filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2));
          transform: scale(1.1);
        }

        .sidebar-icon::after {
          content: attr(title);
          position: absolute;
          left: 100%;
          top: 50%;
          transform: translateY(-50%);
          background: rgba(0, 0, 0, 0.9);
          color: white;
          padding: 8px 12px;
          border-radius: 6px;
          font-size: 14px;
          white-space: nowrap;
          opacity: 0;
          visibility: hidden;
          transition: all 0.3s ease;
          margin-left: 10px;
          z-index: 1000;
          backdrop-filter: blur(10px);
        }

        .sidebar-icon::after {
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .sidebar-icon:hover::after {
          opacity: 1;
          visibility: visible;
          margin-left: 15px;
        }

        @keyframes slideInLeft {
          from {
            transform: translateX(-100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }

        @keyframes bounceIn {
          0% {
            transform: scale(0.3);
            opacity: 0;
          }
          50% {
            transform: scale(1.05);
          }
          70% {
            transform: scale(0.9);
          }
          100% {
            transform: scale(1);
            opacity: 1;
          }
        }

        .sidebar-icon:nth-child(1) {
          animation: bounceIn 0.6s ease-out 0.1s both;
        }
        .sidebar-icon:nth-child(2) {
          animation: bounceIn 0.6s ease-out 0.2s both;
        }
        .sidebar-icon:nth-child(3) {
          animation: bounceIn 0.6s ease-out 0.3s both;
        }
        .sidebar-icon:nth-child(4) {
          animation: bounceIn 0.6s ease-out 0.4s both;
        }
        .sidebar-icon:nth-child(5) {
          animation: bounceIn 0.6s ease-out 0.5s both;
        }

        /* Enhance existing styles without breaking them */
        .sidebar-icon {
          background: linear-gradient(
            135deg,
            rgba(255, 255, 255, 0.1),
            rgba(255, 255, 255, 0.05)
          );
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
      `}</style>

      <nav className="sidebar">
        <div className="sidebar-icon" title="Restaurants">
          <PiForkKnifeFill className="text-2xl" />
        </div>
        <div className="sidebar-icon" title="Hotels">
          <MdHotel className="text-2xl" />
        </div>
        <div
          className="sidebar-icon"
          title="Plan Route"
          onClick={() => navigate("/route")}
        >
          <FaRoute className="text-2xl" />
        </div>
        <div
          className="sidebar-icon"
          title="Find Restaurants Along Route"
          onClick={() => navigate("/route-planner")}
        >
          <MdRestaurant className="text-2xl" />
        </div>
        <div
          className="sidebar-icon"
          title="Set Preferences"
          onClick={() => setIsPreferencesOpen(true)}
        >
          <MdSettings className="text-2xl" />
        </div>
        <div
          className="sidebar-icon"
          title="Review Restaurant"
          onClick={() => setIsReviewsOpen(true)}
        >
          <MdOutlineMuseum className="text-2xl" />
        </div>
        <div
          className="sidebar-icon"
          title="Entertainment"
          onClick={() => setIsEntertainmentOpen(true)}
        >
          <MdTheaters className="text-2xl" />
        </div>
      </nav>

      <Modal
        isOpen={isPreferencesOpen}
        onClose={() => setIsPreferencesOpen(false)}
      >
        <UserPreferencePage onClose={() => setIsPreferencesOpen(false)} />
      </Modal>

      <UserReviewsPage
        isOpen={isReviewsOpen}
        onClose={() => setIsReviewsOpen(false)}
      />

      <EntertainmentPage
        isOpen={isEntertainmentOpen}
        onClose={() => setIsEntertainmentOpen(false)}
      />
    </>
  );
};

export default Items;

