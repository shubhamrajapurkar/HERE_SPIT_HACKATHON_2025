import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ScrollView,
  TouchableHighlight,
  TouchableOpacity,
  Alert,
} from "react-native";
import { FontAwesome5 } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import CarbonSave from "../carbonsave";
import { useFocusEffect } from "@react-navigation/native"; // For navigation focus handling

const RouteCard = ({ id, route }) => {
  const [showLeftFade, setShowLeftFade] = useState(false);
  const [showRightFade, setShowRightFade] = useState(true);
  const [selectedLeg, setSelectedLeg] = useState(null);

  useEffect(() => {
    // console.log("RouteCard", route);
  }, [route]);
  const modeIcons = {
    WALK: <FontAwesome5 name="walking" size={20} color="#065f46" />, // Slightly smaller icon size
    BUS: <FontAwesome5 name="bus" size={20} color="#065f46" />,
    RAIL: <FontAwesome5 name="train" size={20} color="#065f46" />,
    SUBWAY: <FontAwesome5 name="subway" size={20} color="#065f46" />,
  };

  // Calculate total duration and reach-by time
  const totalTimeInSeconds = route.duration;
  const totalTimeInMinutes = Math.round(totalTimeInSeconds / 60);
  const reachByTime = new Date(
    Date.now() + totalTimeInSeconds * 1000
  ).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  // Calculate time for each mode
  const legTimes = route.legs.map((leg) => {
    const startTime = new Date(leg.startTime);
    const endTime = new Date(leg.endTime);
    return {
      mode: leg.mode,
      timeInMinutes: Math.round((endTime - startTime) / 60000), // Convert ms to minutes
    };
  });

  const handleScroll = (event) => {
    const { contentOffset, contentSize, layoutMeasurement } = event.nativeEvent;

    // Check if scrolling reached the beginning or end
    setShowLeftFade(contentOffset.x > 0);
    setShowRightFade(
      contentOffset.x < contentSize.width - layoutMeasurement.width
    );
  };

  const handleIconPress = (index, mode) => {
    console.log(index, mode);

    if (mode === "RAIL" || mode === "SUBWAY") {
      Alert.alert(
        "View Railway Timetable",
        "Would you like to view the railway timetable?",
        [
          {
            text: "Cancel",
            style: "cancel",
          },
          {
            text: "OK",
            onPress: () => {
              router.push("/(tabs)/travel/timetable");
            },
          },
        ]
      );
    }
  };

  return (
    <View className="mb-4 bg-white rounded-2xl shadow-md overflow-hidden">
      {/* Top Section with Mode Icons */}
      <View className="p-4 bg-gray-50">
        <View className="relative">
          {showLeftFade && (
            <LinearGradient
              colors={["rgba(236,253,245,1)", "rgba(236,253,245,0)"]}
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
                <TouchableOpacity
                  onPress={() => handleIconPress(index, leg.mode)}
                  className="bg-white p-3 rounded-xl shadow-sm"
                >
                  <View className="items-center">
                    {modeIcons[leg.mode]}
                    <Text className="text-xs text-gray-600 mt-2 font-medium">
                      {leg.timeInMinutes}m
                    </Text>
                  </View>
                </TouchableOpacity>
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
              colors={["rgba(236,253,245,0)", "rgba(236,253,245,1)"]}
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
                  {totalTimeInMinutes}
                </Text>
                <Text className="text-sm text-gray-600 ml-1">min</Text>
              </View>
            </View>
          </View>
          <CarbonSave value={route.carbonEmission} />
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
                  ₹{route.totalCost}
                </Text>
              </View>
            </View>
          </View>

          <TouchableOpacity
            onPress={() => {
              router.push({
                pathname: "/(tabs)/travel/mapscreen",
                params: { id: id, route: JSON.stringify(route) },
              });
            }}
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

export default RouteCard;
