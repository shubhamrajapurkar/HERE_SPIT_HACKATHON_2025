import React, { useState, useEffect } from "react";
import {
  ScrollView,
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
  TouchableOpacity,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import RouteCardHere from "../../../components/ui/RouteCardHere";
import { useLocalSearchParams } from "expo-router";
import axios from "axios";
import RouteCardHere from "../../../components/ui/RouteCardHere";

const addCarbonEmissions = (data) => {
  // Carbon emission rates in kg CO2/km
  const carbonEmissionRates = {
    WALK: 0,
    RAIL: 0.041,
    SUBWAY: 0.036,
    BUS: 0.089,
  };
  // Function to calculate emissions for a single itinerary
  const calculateCarbonEmissions = (itinerary) => {
    let totalEmissions = 0;

    // Iterate through each leg of the journey
    itinerary.legs.forEach((leg) => {
      // Get the mode of transport
      const mode = leg.mode;
      // Convert distance from meters to kilometers
      const distanceKm = leg.distance / 1000;
      // Calculate emissions for this leg
      const emissions = distanceKm * carbonEmissionRates[mode];
      totalEmissions += emissions;
    });

    return Number(totalEmissions.toFixed(3)); // Round to 3 decimal places
  };

  // Calculate and add carbonEmission for each itinerary
  data.plan.itineraries.forEach((itinerary) => {
    itinerary.carbonEmission = calculateCarbonEmissions(itinerary);
  });

  return data;
};

const addItineraryCosts = (data) => {
  // Cost per km for each mode of transport
  const modeCosts = {
    WALK: 0, // Walking is free
    RAIL: 5, // Train/Rail costs ₹5 per km
    SUBWAY: 4, // Subway costs ₹4 per km
    BUS: 6, // Bus costs ₹6 per km
  };

  // Function to calculate total cost for a single itinerary
  const calculateItineraryCost = (itinerary) => {
    let totalCost = 0;

    // Iterate through each leg of the journey
    itinerary.legs.forEach((leg) => {
      // Get the mode of transport
      const mode = leg.mode;
      // Convert distance from meters to kilometers
      const distanceKm = leg.distance / 1000;
      // Calculate cost for this leg
      const cost = distanceKm * modeCosts[mode];
      totalCost += cost;
    });

    return Number(totalCost.toFixed(0)); // Round to the nearest integer
  };

  // Calculate and add totalCost for each itinerary
  data.plan.itineraries.forEach((itinerary) => {
    itinerary.totalCost = calculateItineraryCost(itinerary);
  });

  return data;
};

const RoutesScreen = () => {
  console.log(process.env.EXPO_PUBLIC_machine_ip);
  const [selectedTab, setSelectedTab] = useState("greenest");
  const { startLatLng, endLatLng } = useLocalSearchParams();
  const [isLoading, setIsLoading] = useState(true);
  const [routes, setRoutes] = useState([]);
  const [error, setError] = useState(null);
  const [carRoute, setCarRoute] = useState([]);

  // Parse coordinates safely
  let startLat, startLng, endLat, endLng;
  try {
    [startLat, startLng] = JSON.parse(startLatLng || "[]");
    [endLat, endLng] = JSON.parse(endLatLng || "[]");
  } catch {
    setError("Invalid coordinates provided.");
    return null;
  }

  const API_URL = `http://${process.env.EXPO_PUBLIC_machine_ip}:8080/otp/routers/default/plan?fromPlace=${startLat}%2C${startLng}&toPlace=${endLat}%2C${endLng}&date=2025-01-16&time=09:00:00&arriveBy=false&mode=TRANSIT%2CWALK&maxWalkDistance=1000&numItineraries=10`;
  console.log(API_URL);

  let resData;
  useEffect(() => {
    const fetchRoutes = async () => {
      try {
        const response = await axios.get(API_URL);

        // HERE Routing API for car route
        const hereUrl = `https://router.hereapi.com/v8/routes?transportMode=car&origin=${startLat},${startLng}&destination=${endLat},${endLng}&return=summary,polyline&apikey=${hereApiKey}`;

        const hereResponse = await axios.get(hereUrl);

        setCarRoute(hereResponse.data);

        resData = addItineraryCosts(addCarbonEmissions(response.data));
        if (resData?.plan?.itineraries) {
          setRoutes(resData.plan.itineraries);
          // Log the complete routes data after processing
          console.log("Complete Routes Data:", JSON.stringify({
            numberOfRoutes: resData.plan.itineraries.length,
            routes: resData.plan.itineraries
          }, null, 2));
        } else {
          throw new Error("No itineraries found in the response.");
        }
      } catch (err) {
        console.error("Error fetching routes:", err);
        setError("Failed to load routes. Please try again later.");
      } finally {
        setIsLoading(false);
      }
    };

    // Ensure coordinates are valid before fetching
    if (startLat && startLng && endLat && endLng) {
      fetchRoutes();
    } else {
      setError("Missing or invalid start/end locations.");
      setIsLoading(false);
    }
  }, [startLat, startLng, endLat, endLng]);

  useEffect(() => {
    if (selectedTab === "greenest") {
      setRoutes((prevRoutes) =>
        [...prevRoutes].sort((a, b) => a.carbonEmission - b.carbonEmission)
      );
    }
    if (selectedTab === "quickest") {
      setRoutes((prevRoutes) =>
        [...prevRoutes].sort((a, b) => a.duration - b.duration)
      );
    }
    if (selectedTab === "cheapest") {
      setRoutes((prevRoutes) =>
        [...prevRoutes].sort((a, b) => a.totalCost - b.totalCost)
      );
    }
  }, [selectedTab]);

  if (isLoading) {
    return (
      <View className="flex-1 justify-center items-center bg-gray-50 p-4">
        <ActivityIndicator size="large" color="#047857" />
        <Text className="mt-2 text-base text-gray-600">Loading routes...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View className="flex-1 justify-center items-center bg-gray-50 p-4">
        <Text className="text-base text-red-600">{error}</Text>
      </View>
    );
  }

  return (
    <ScrollView className="flex-1 bg-gray-100 px-4 pt-4">
      {/* Tab Container */}
      <View className="mb-6 bg-white rounded-2xl shadow-sm overflow-hidden border border-gray-100">
        <View className="flex-row p-3">
          <TouchableOpacity
            onPress={() => setSelectedTab("greenest")}
            className={`px-4 py-3 rounded-xl flex-row items-center justify-center ${
              selectedTab === "greenest" ? "bg-primary" : "bg-gray-50/80"
            } flex-1 mx-1`}
          >
            <Ionicons 
              name="leaf" 
              size={18} 
              color={selectedTab === "greenest" ? "#ffffff" : "#059669"} 
              style={{ marginRight: 8 }}
            />
            <Text
              className={`text-base font-semibold ${
                selectedTab === "greenest" ? "text-white" : "text-primary"
              }`}
            >
              Greenest
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => setSelectedTab("quickest")}
            className={`px-4 py-3 rounded-xl flex-row items-center justify-center ${
              selectedTab === "quickest" ? "bg-primary" : "bg-gray-50"
            } flex-1 mx-1`}
          >
            <Ionicons 
              name="flash" 
              size={18} 
              color={selectedTab === "quickest" ? "#ffffff" : "#047857"} 
              style={{ marginRight: 6 }}
            />
            <Text
              className={`text-base font-medium ${
                selectedTab === "quickest" ? "text-white" : "text-primary"
              }`}
            >
              Quickest
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => setSelectedTab("cheapest")}
            className={`px-4 py-3 rounded-xl flex-row items-center justify-center ${
              selectedTab === "cheapest" ? "bg-primary" : "bg-gray-50"
            } flex-1 mx-1`}
          >
            <Ionicons 
              name="wallet" 
              size={18} 
              color={selectedTab === "cheapest" ? "#ffffff" : "#047857"} 
              style={{ marginRight: 6 }}
            />
            <Text
              className={`text-base font-medium ${
                selectedTab === "cheapest" ? "text-white" : "text-primary"
              }`}
            >
              Cheapest
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Routes Section */}
      {selectedTab === "quickest" && (
        <RouteCardHere route={carRoute} />
      )}
      
      {routes.map((route, idx) => (
        <RouteCardHere
          key={idx}
          id={idx}
          route={route}
        />
      ))}

      {!(selectedTab === "quickest") && (
        <RouteCardHere route={carRoute} />
      )}

      {routes.length === 0 && (
        <Text className="text-base text-red-600 text-center mt-4">
          No routes available.
        </Text>
      )}
    </ScrollView>
  );
};

// Remove StyleSheet since we're using Tailwind classes
export default RoutesScreen;
