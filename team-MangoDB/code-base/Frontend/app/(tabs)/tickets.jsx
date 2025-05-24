import React, { useEffect, useState } from "react";
import { SafeAreaView, View, Text, ScrollView } from "react-native";
import { FontAwesome5 } from "@expo/vector-icons";
import QRCode from "react-native-qrcode-svg";
import { useLocalSearchParams } from "expo-router";

const TicketPage = () => {
  const { legz } = useLocalSearchParams();
  const [legs, setLegs] = useState([]);

  useEffect(() => {
    if (legz) {
      try {
        const parsedLegs = JSON.parse(legz);
        setLegs(parsedLegs);
      } catch (error) {
        console.error("Error parsing legs:", error);
        setLegs([]);
      }
    }
  }, [legz]);

  const transportLegs = legs.filter(
    (leg) => ["RAIL", "SUBWAY", "BUS"].includes(leg.mode)
  );

  const getDuration = (start, end) => {
    const startTime = new Date(start);
    const endTime = new Date(end);
    const durationInMinutes = Math.floor((endTime - startTime) / (1000 * 60));
    const hours = Math.floor(durationInMinutes / 60);
    const minutes = durationInMinutes % 60;
    
    return hours > 0 
      ? `${hours}h ${minutes}m` 
      : `${minutes}m`;
  };

  const formatTime = (isoString) => {
    return new Date(isoString).toLocaleTimeString('en-IN', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true
    });
  };

  const getValidTillTime = (startTime, mode) => {
    if (mode !== "SUBWAY") {
      return new Date(new Date(startTime).getTime() + 3600000); // 1 hour validity for non-subway
    }
  
    // Get CURRENT DATE in local time (device timezone)
    const now = new Date();
    
    // Set to 1 AM NEXT DAY
    const nextDay = new Date(
      now.getFullYear(),
      now.getMonth(),
      now.getDate() + 1, // This automatically handles month/year rollover
      1, 0, 0 // 1:00 AM
    );
  
    return nextDay;
  };
  
  const formatDate = (date) => {
    // Simple DD/MM formatting
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    return `${day}/${month}`;
  };

  const modeStyles = {
    BUS: { icon: "bus", color: "#047857" },
    SUBWAY: { icon: "subway", color: "#047857" },
    RAIL: { icon: "train", color: "#047857" },
    WALK: { icon: "walking", color: "#047857" }
  };  

  return (
    <SafeAreaView className="flex-1 bg-gray-50">
      {/* Header */}
      {/* <View className="bg-primary px-4 py-5 z-10 shadow-sm">
        <Text className="text-2xl pt-9 font-bold text-white text-center">Your Tickets</Text>
        <Text className="text-emerald-100 text-center mt-1">
          {transportLegs.length} active tickets
        </Text>
      </View> */}
  
      <ScrollView className="py-3 -mt-3">
        <View className="bg-white rounded-t-3xl py-6 px-4">
          {transportLegs.map((leg, index) => (
            <View key={index} className="bg-white rounded-xl p-4 mb-4 shadow-sm border border-gray-100">
              {/* Top Section */}
              <View className="flex-row justify-between items-center pb-3 border-b border-gray-100">
                <View className="flex-row items-center space-x-3">
                  <FontAwesome5
                    name={modeStyles[leg.mode].icon}
                    size={20}
                    color="#047857"
                  />
                  <Text className="py-1 pl-2 text-lg font-semibold text-gray-800">
                    {leg.mode == "BUS" ? `BEST ${leg.routeShortName}` : leg.route}
                    {leg.mode === "RAIL" && " Railway"}
                  </Text>
                </View>
                <View className="bg-emerald-50 px-3 py-1 rounded-md">
                  <Text className="text-primary text-sm font-medium">
                    {getDuration(leg.startTime, leg.endTime)}
                  </Text>
                </View>
              </View>
  
              {/* First Part: Station Info + QR Code */}
              <View className="pt-4 flex-row justify-between items-start">
                {/* Station Details */}
                <View className="flex-1 pr-4">
                  <View className="space-y-4">
                    <View className="flex-row items-center space-x-3">
                      <View className="w-2 h-2 rounded-full bg-emerald-500" />
                      <Text className="pl-2 text-gray-700 flex-1" numberOfLines={2}>
                        {leg.from.name}
                      </Text>
                    </View>
                    <View className="ml-1 h-8 border-l-2 border-dashed border-emerald-200" />
                    <View className="flex-row items-center space-x-3">
                      <View className="w-2 h-2 rounded-full bg-emerald-700" />
                      <Text className="pl-2 text-gray-700 flex-1" numberOfLines={2}>
                        {leg.to.name}
                      </Text>
                    </View>
                  </View>

                  {/* Distance Info */}
                  <View className="mt-4 flex-row items-center space-x-2">
                    <View className="bg-emerald-50 p-2 rounded-lg">
                      <FontAwesome5 name="route" size={16} color="#047857" />
                    </View>
                    <Text className="pl-1 text-emerald-700 font-medium">
                      {leg.distance > 1000 
                        ? `${(leg.distance/1000).toFixed(1)} km`
                        : `${Math.round(leg.distance)} m`}
                    </Text>
                  </View>
                </View>

                {/* QR Code */}
                <QRCode
                  value={`${leg.mode}-${leg.from.name}-${leg.to.name}`}
                  size={110}
                  backgroundColor="white"
                  color="#047857"
                />
              </View>

              {/* Second Part: Validity Info - Horizontal Layout */}
              <View className="mt-6 pt-4 border-t border-gray-100 flex-row justify-between">
                <View className="space-y-1">
                  <Text className="text-sm text-gray-500">Issued at</Text>
                  <Text className="text-sm font-medium text-emerald-700">
                    {formatTime(leg.startTime)}
                  </Text>
                </View>
                <View className="space-y-1">
                  <Text className="text-sm text-gray-500">Valid till</Text>
                  <Text className="text-sm font-medium text-emerald-700">
                    {formatTime(getValidTillTime(leg.startTime, leg.mode))}
                    {leg.mode === "SUBWAY" && (
                      <Text> {formatDate(getValidTillTime(leg.startTime, leg.mode))}</Text>
                    )}
                  </Text>
                </View>
              </View>
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default TicketPage;