import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  Keyboard 
} from "react-native";
import React, { useState, useCallback } from "react";
import { SafeAreaView } from "react-native-safe-area-context";
import { MaterialIcons } from "@expo/vector-icons";
import uuid from "react-native-uuid";
import axios from "axios";
import * as Location from "expo-location";
import { router } from "expo-router";

// Debounce utility
const debounce = (func, delay) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => func(...args), delay);
  };
};

const Travel = () => {
  const [fromLocation, setFromLocation] = useState("");
  const [toLocation, setToLocation] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [activeInput, setActiveInput] = useState("");
  const [startLatLng, setStartLatLng] = useState(null);
  const [endLatLng, setEndLatLng] = useState(null);

  const API_KEY = process.env.EXPO_PUBLIC_here_api; // Replace with your actual HERE API key
  const BASE_URL = "https://autosuggest.search.hereapi.com/v1/autosuggest";

  // Fetch location suggestions using HERE API
  const fetchSuggestions = async (input) => {
    try {
      if (!input) {
        setSuggestions([]);
        return;
      }

     // Example: at=52.5200,13.4050 (Berlin)
      const params = {
        q: input,
        apiKey: API_KEY,
        limit: 5,
      };

      const response = await axios.get(BASE_URL, { params });

      // Map HERE API results to match your UI expectations
      const predictions = (response.data.items || []).map((item) => ({
        place_id: item.id,
        description: item.title,
        structured_formatting: {
          secondary_text: item.address?.label || "",
        },
        geometry: {
          location: {
            lat: item.position?.lat,
            lng: item.position?.lng,
          },
        },
      }));

      setSuggestions(predictions);
    } catch (error) {
      console.error("Error fetching suggestions:", error);
    }
  };

  const debouncedFetchSuggestions = useCallback(
    debounce(fetchSuggestions, 300),
    []
  );

  const handleInputChange = (text, inputType) => {
    if (inputType === "from") setFromLocation(text);
    else setToLocation(text);

    setActiveInput(inputType);
    debouncedFetchSuggestions(text);
  };

  const handleClearInput = (inputType) => {
    if (inputType === "from") {
      setFromLocation("");
      setStartLatLng(null);
    } else {
      setToLocation("");
      setEndLatLng(null);
    }
    setSuggestions([]);
  };

  const handleSelectSuggestion = (item) => {
    if (activeInput === "from") {
      setFromLocation(item.description);
      setStartLatLng([item.geometry.location.lat, item.geometry.location.lng]);
    } else {
      setToLocation(item.description);
      setEndLatLng([item.geometry.location.lat, item.geometry.location.lng]);
    }
    setSuggestions([]);
  };

  const fetchCurrentLocation = async (inputType) => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") {
        alert("Permission to access location was denied.");
        return;
      }
      const location = await Location.getCurrentPositionAsync({});
      const { latitude, longitude } = location.coords;

      if (inputType === "from") {
        setStartLatLng([latitude, longitude]);
        setFromLocation("Current Location");
      } else {
        setEndLatLng([latitude, longitude]);
        setToLocation("Current Location");
      }
    } catch (error) {
      console.error("Error fetching location:", error);
      alert("Failed to fetch current location.");
    }
  };

  return (
    <SafeAreaView
      className="min-h-full bg-background px-4 py-4"
      edges={["left", "right"]}
    >
      <View className="bg-white shadow-lg rounded-lg mb-4 p-6">
        {/* From Section */}
        <View className="mb-4 relative">
          <Text
            className="absolute -top-3 left-4 bg-white px-1 text-gray-600 font-psemibold text-sm"
            style={{ zIndex: 1 }}
          >
            From
          </Text>
          <View className="flex-row items-center border border-gray-300 rounded-md px-3 py-1.5">
            <MaterialIcons
              name="location-on"
              size={20}
              color="#065f46"
              className="mr-3"
            />
            <TextInput
              placeholder="Starting location"
              value={fromLocation}
              onChangeText={(text) => handleInputChange(text, "from")}
              className="flex-1 text-gray-800 text-sm font-pmedium"
              style={{ height: 40 }}
            />
            {fromLocation !== "" && (
              <TouchableOpacity onPress={() => handleClearInput("from")}>
                <MaterialIcons name="close" size={20} color="#888" />
              </TouchableOpacity>
            )}
            <TouchableOpacity
              onPress={() => fetchCurrentLocation("from")}
              className="ml-2"
            >
              <MaterialIcons name="gps-fixed" size={20} color="#065f46" />
            </TouchableOpacity>
          </View>
        </View>

        {/* To Section */}
        <View className="relative">
          <Text
            className="absolute -top-3 left-4 bg-white px-1 text-gray-600 font-psemibold text-sm"
            style={{ zIndex: 1 }}
          >
            To
          </Text>
          <View className="flex-row items-center border border-gray-300 rounded-md px-3 py-1.5">
            <MaterialIcons
              name="location-on"
              size={20}
              color="#065f46"
              className="mr-3"
            />
            <TextInput
              placeholder="Destination"
              value={toLocation}
              onChangeText={(text) => handleInputChange(text, "to")}
              className="flex-1 text-gray-800 text-sm font-pmedium"
              style={{ height: 40 }}
            />
            {toLocation !== "" && (
              <TouchableOpacity onPress={() => handleClearInput("to")}>
                <MaterialIcons name="close" size={20} color="#888" />
              </TouchableOpacity>
            )}
            <TouchableOpacity
              onPress={() => fetchCurrentLocation("to")}
              className="ml-2"
            >
              <MaterialIcons name="gps-fixed" size={20} color="#065f46" />
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/* Suggestions List */}
      {suggestions.length > 0 ? (
        <FlatList
          keyboardShouldPersistTaps="handled"
          data={suggestions}
          keyExtractor={(item) => item.place_id}
          renderItem={({ item }) => (
            <TouchableOpacity
              onPress={() => {
                Keyboard.dismiss();
                handleSelectSuggestion(item);
              }}
              className="flex-row items-center bg-white shadow-sm rounded-lg px-4 mx-1 mb-2 h-16"
            >
              <MaterialIcons name="place" size={24} color="#065f46" />
              <View className="ml-4 flex-1">
                <Text
                  className="font-medium text-gray-900 text-base truncate"
                  numberOfLines={1}
                >
                  {item.description}
                </Text>
                <Text
                  className="text-gray-500 text-sm truncate"
                  numberOfLines={1}
                >
                  {item.structured_formatting?.secondary_text}
                </Text>
              </View>
            </TouchableOpacity>
          )}
          contentContainerStyle={{ paddingBottom: 64 }}
        />
      ) : (
        <Text className="text-gray-500 text-center mt-4 font-psemibold">
          No suggestions available. Try searching for another location.
        </Text>
      )}

      {/* Navigate Button */}
      <TouchableOpacity
        disabled={!startLatLng || !endLatLng}
        className={`absolute bottom-[1rem] left-4 right-4 py-3 shadow-lg rounded-full ${
          startLatLng && endLatLng ? "bg-primary" : "bg-gray-400"
        }`}
        onPress={() => {
          router.push({
            pathname: "/(tabs)/travel/routescreen",
            params: {
              startLatLng: JSON.stringify(startLatLng),
              endLatLng: JSON.stringify(endLatLng),
            },
          });
        }}
      >
        <Text className="text-white text-center font-bold text-lg">
          Find Routes
        </Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
};

export default Travel;
