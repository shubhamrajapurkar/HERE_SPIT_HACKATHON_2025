import { useState, useEffect } from "react";
import HereMap from "./HereMap";
import SearchBox from "./SearchBox";
import { FaBars } from "react-icons/fa";
import Menu from "./Menu";
import Items from "./Items";

const Map = () => {
  const [position, setPosition] = useState(null);
  const [openMenu, setOpenMenu] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [showRestaurants, setShowRestaurants] = useState(false);

  // Fetch user location or fall back
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setPosition({
            lat: pos.coords.latitude,
            lng: pos.coords.longitude,
          });
          setIsLoading(false);
        },
        () => {
          setPosition({ lat: 19.1152154, lng: 72.8424089 }); // London fallback
          setIsLoading(false);
        }
      );
    } else {
      setPosition({ lat: 19.1152154, lng: 72.8424089 });
      setIsLoading(false);
    }
  }, []);

  if (isLoading || !position) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-gray-100">
        <div className="p-6 bg-white rounded-lg shadow-lg">
          <p>Loading map...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Top bar with Menu, SearchBox */}
      <div className="flex md:absolute items-center md:top-[4px] left-[40%] relative md:left-[50%] translate-x-[-50%] z-10 pl-16 md:pl-0">
        <div
          onClick={() => setOpenMenu((prev) => !prev)}
          className="text-[#424242] text-[19px] bg-white p-3 rounded-full hover:bg-gray-100 cursor-pointer"
          title="Menu"
        >
          <FaBars />
        </div>

        <div className="md:w-[300px] w-[350px] md:shadow-md cursor-text">
          <SearchBox setPosition={setPosition} position={position} />
        </div>

        <div
          className="text-[#424242] text-[19px] bg-white p-[10px] shadow-none md:shadow-md cursor-pointer"
          title="Search"
        >

        </div>
      </div>

      {/* Toggle filters like showRestaurants */}
      <Items
        showRestaurants={showRestaurants}
        setShowRestaurants={setShowRestaurants}
      />

      {/* Side menu */}
      <Menu openMenu={openMenu} setOpenMenu={setOpenMenu} />

      {/* Map Container */}
      <div className="relative flex-1">
        <HereMap
          zoom={17}
          position={position}
          showRestaurants={showRestaurants}
        />
      </div>
    </>
  );
};

export default Map;
