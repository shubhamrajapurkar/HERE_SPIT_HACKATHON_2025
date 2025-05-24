import React, { useState } from "react";
import { 
  View, 
  Text, 
  ScrollView, 
  TouchableHighlight,
  TouchableOpacity 
} from "react-native";
import { FontAwesome5 } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import CarbonSave from "../carbonsave";

const RouteCardHere = ({ route }) => {
  const [showLeftFade, setShowLeftFade] = useState(false);
  const [showRightFade, setShowRightFade] = useState(true);

  const modeIcons = {
    WALK: <FontAwesome5 name="walking" size={22} color="#065f46" />,
    BUS: <FontAwesome5 name="bus" size={22} color="#065f46" />,
    RAIL: <FontAwesome5 name="train" size={22} color="#065f46" />,
    SUBWAY: <FontAwesome5 name="subway" size={22} color="#065f46" />,
  };

  // Extract route details
  const { legs, overview_polyline } = route.routes[0];
  console.log(legs);
  const totalDistance = legs.reduce((sum, leg) => sum + leg.distance, 0);
  const carbonValue = (0.192 * totalDistance) / 1000;
  const totalDuration = legs.reduce((sum, leg) => sum + leg.duration, 0);

  const reachByTime = new Date(
    Date.now() + totalDuration * 1000
  ).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  const calculateCabFare = (response) => {
  
    const baseFare = 50; // ₹50 base fare
    const costPerKm = 12; // ₹12 per km

    // Extract distance from the response
    const totalDistanceMeters = response.routes[0].legs[0].distance; // In meters
    const totalDistanceKm = totalDistanceMeters / 1000; // Convert to kilometers

    // Calculate total fare
    const totalFare = baseFare + totalDistanceKm * costPerKm;

    // Return the rounded total fare
    return Math.round(totalFare);
  };

  // Example Usage
  const response = {
    // Include your provided response JSON here
    routes: [
      {
        legs: [
          {
            distance: 3836, // Example distance in meters
          },
        ],
      },
    ],
  };

  const totalFare = calculateCabFare(route);
  // Extract time and mode for each leg
  const legTimes = legs.map((leg) => ({
    mode: "DRIVE", // Assuming all legs are by car; adjust if data includes mode.
    timeInMinutes: Math.round(leg.duration / 60),
    readableDistance: leg.readable_distance,
    instructions: leg.steps.map((step) => step.instructions).join(", "),
    cost: Number(leg.readable_distance * 15).toFixed(0),
  }));

  const handleScroll = (event) => {
    const { contentOffset, contentSize, layoutMeasurement } = event.nativeEvent;

    setShowLeftFade(contentOffset.x > 0);
    setShowRightFade(
      contentOffset.x < contentSize.width - layoutMeasurement.width
    );
  };

  return (
    <View className="mb-4 bg-white rounded-2xl shadow-sm overflow-hidden border border-gray-100">
      {/* Top Section with Mode Icons */}
      <View className="p-4 bg-white">
        <View className="relative">
          {showLeftFade && (
            <LinearGradient
              colors={["rgba(255,255,255,1)", "rgba(255,255,255,0)"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                bottom: 0,
                width: 30,
                zIndex: 1,
              }}
            />
          )}

          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{
              flexDirection: "row",
              alignItems: "center",
              paddingHorizontal: 4,
              gap: 16,
            }}
            onScroll={handleScroll}
            scrollEventThrottle={16}
          >
            {legTimes.map((leg, index) => (
              <View key={index} className="flex-row items-center">
                <View className="bg-gray-50 p-3 rounded-xl border border-gray-100">
                  <View className="items-center">
                    <FontAwesome5 name="car" size={20} color="#047857" />
                    <Text className="text-xs text-gray-600 mt-2 font-medium">
                      {leg.timeInMinutes}m
                    </Text>
                  </View>
                </View>
                {index < legTimes.length - 1 && (
                  <View className="mx-2">
                    <FontAwesome5
                      name="long-arrow-alt-right"
                      size={16}
                      color="#047857"
                    />
                  </View>
                )}
              </View>
            ))}
          </ScrollView>

          {showRightFade && (
            <LinearGradient
              colors={["rgba(255,255,255,0)", "rgba(255,255,255,1)"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={{
                position: "absolute",
                right: 0,
                top: 0,
                bottom: 0,
                width: 30,
                zIndex: 1,
              }}
            />
          )}
        </View>
      </View>

      {/* Info Section */}
      <View className="p-4">
        <View className="flex-row justify-between items-center mb-4">
          <View className="flex-row items-center space-x-3">
            <View className="bg-gray-100 p-2.5 rounded-lg">
              <FontAwesome5 name="clock" size={16} color="#047857" />
            </View>
            <View className="ml-2">
              <Text className="text-xs text-gray-500 mb-[-2px]">Total Duration</Text>
              <View className="flex-row items-baseline">
                <Text className="text-lg font-bold text-gray-800">
                  {Math.round(totalDuration / 60)}
                </Text>
                <Text className="text-sm text-gray-600 ml-1">min</Text>
              </View>
            </View>
          </View>
          <CarbonSave value={Number(carbonValue.toFixed(3)) || 0} />
        </View>

        <View className="flex-row items-center justify-between">
          <View className="flex-row items-center space-x-3">
            <View className="bg-gray-100 p-2.5 rounded-lg">
              <FontAwesome5 name="wallet" size={16} color="#047857" />
            </View>
            <View className="ml-2">
              <Text className="text-xs text-gray-500 mb-[-2px]">Total Fare</Text>
              <View className="flex-row items-baseline">
                <Text className="text-lg font-bold text-gray-800">
                  ₹{totalFare || "80"}
                </Text>
              </View>
            </View>
          </View>

          <TouchableOpacity
            onPress={() =>
              router.push({
                pathname: "/(tabs)/travel/mapscreen",
                params: { overview_polyline },
              })
            }
            className="bg-gray-100 px-5 py-2.5 rounded-lg border border-primary"
            activeOpacity={0.7}
          >
            <Text className="text-primary font-medium">View Route</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

export default RouteCardHere;
