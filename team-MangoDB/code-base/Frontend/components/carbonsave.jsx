import React from "react";
import { View, Text } from "react-native";
import { FontAwesome5 } from "@expo/vector-icons";

const CarbonSave = ({ value }) => {
  return (
    <View className="flex-row items-center px-2 py-1 rounded-lg bg-gray-100 self-start">
      <FontAwesome5 name="leaf" size={16} color="#047857" className="mr-2" />
      <View>
        <Text className="text-xs font-medium text-primary">Carbon Save</Text>
        <Text className="text-sm font-bold text-primary">
          {value ? `${value} kg` : "0 kg"}
        </Text>
      </View>
    </View>
  );
};

export default CarbonSave;
