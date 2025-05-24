import React, { useEffect, useState } from "react";
import { View, Text, TouchableOpacity, Alert, ScrollView, SafeAreaView } from "react-native";
import { useLocalSearchParams, router } from "expo-router";
import { Linking } from "react-native";
import { MaterialIcons, FontAwesome } from "@expo/vector-icons";

const BookingPage = () => {
    const { selectedLegs, amount } = useLocalSearchParams();
    const [legs, setLegs] = useState([]);
    const [bookingAmount, setBookingAmount] = useState(null);

    useEffect(() => {
        if (selectedLegs) {
            setLegs(JSON.parse(selectedLegs));
        }
        if (amount) {
            setBookingAmount(amount);
        }
    }, [selectedLegs, amount]);

    const formatDuration = (duration) => {
        const minutes = Math.round(duration / 60);
        return `${minutes} min`;
    };

    const renderIcon = (mode) => {
        switch (mode) {
            case "WALK":
                return <FontAwesome name="male" size={24} color="#047857" />;
            case "RAIL":
                return <MaterialIcons name="train" size={24} color="#047857" />;
            case "BUS":
                return <FontAwesome name="bus" size={24} color="#047857" />;
            case "SUBWAY":
                return <FontAwesome name="subway" size={24} color="#047857" />;
            default:
                return <FontAwesome name="question" size={24} color="#047857" />;
        }
    };

    const renderLegCard = (leg, index) => (
        <View
            key={index}
            className="bg-white rounded-lg p-4 mb-3 shadow-sm border border-gray-100"
        >
            {/* Transport Mode Header */}
            <View className="flex-row items-center justify-between mb-2">
                <View className="flex-row items-center space-x-3">
                    <View className="p-2 bg-emerald-100 rounded-md">
                        {renderIcon(leg.mode)}
                    </View>
                    <Text className="pl-3 text-lg font-semibold text-gray-800">
                        {leg.mode == "BUS" ? `BEST ${leg.routeShortName}` : leg.route}
                        {leg.mode === "RAIL" && " Railway"}
                    </Text>
                </View>
                <View className="bg-emerald-50 px-2.5 py-1 rounded-md">
                    <Text className="text-emerald-700 text-sm font-medium">
                        {formatDuration(leg.duration)}
                    </Text>
                </View>
            </View>

            {/* Route Details */}
            <View className="pl-2 space-y-3">
                <View className="flex-row items-center">
                    <View className="w-2 h-2 rounded-full bg-emerald-500 mr-3" />
                    <Text className="text-gray-700 flex-1" numberOfLines={1}>
                        {leg.from.name}
                    </Text>
                </View>
                
                <View className="ml-1 h-8 border-l-2 border-dashed border-emerald-200" />
                
                <View className="flex-row items-center">
                    <View className="w-2 h-2 rounded-full bg-emerald-700 mr-3" />
                    <Text className="text-gray-700 flex-1" numberOfLines={1}>
                        {leg.to.name}
                    </Text>
                </View>
            </View>

            {/* Distance Footer */}
            <View className="mt-4 border-t border-gray-100 pt-3">
                <View className="flex-row justify-between items-center">
                    <Text className="text-sm text-gray-600">Distance</Text>
                    <Text className="text-emerald-700 font-medium">
                        {leg.distance > 1000 
                            ? `${(leg.distance/1000).toFixed(1)} km`
                            : `${Math.round(leg.distance)} m`}
                    </Text>
                </View>
            </View>
        </View>
    );

    const handlePayment = () => {
        const upiID = "deeppatel223204@okicici";
        const payeeName = "Deep Patel";
        const transactionRef = `TXN_${Date.now()}`;
        const transactionNote = "Booking Payment";
        const sendingAmt = bookingAmount || "0.00";

        const paymentUrl = `upi://pay?pa=${upiID}&pn=${payeeName}&am=${sendingAmt}&cu=INR&tn=${transactionNote}&tr=${transactionRef}`;

        Linking.canOpenURL(paymentUrl)
            .then((supported) => {
                if (!supported) {
                    Alert.alert(
                        "Error",
                        "No UPI apps installed on your device. Please install one and try again."
                    );
                } else {
                    return Linking.openURL(paymentUrl);
                }
            })
            .then(() => {
                setTimeout(() => {
                    router.push({
                        pathname: "../tickets",
                        params: { legz: JSON.stringify(legs) }
                    });
                    // router.replace("/(tabs)/travel/index");
                }, 500);
            })
            .catch((err) => {
                Alert.alert("Error", "Failed to initiate UPI payment.");
            });
    };

    return (
        <SafeAreaView className="flex-1 bg-gray-50">

            <ScrollView className="flex-1 -mt-3">
                <View className="bg-white rounded-t-3xl pt-6 px-4">
                    {/* Journey Cards */}
                    {legs.map((leg, index) => renderLegCard(leg, index))}
                    
                    {/* Payment Section */}
                    <View className="bg-emerald-50 rounded-xl p-4 mb-6 mt-4 border border-emerald-100">
                        <View className="flex-row justify-between items-center">
                            <View>
                                <Text className="text-sm text-emerald-800">Total Fare</Text>
                                <Text className="text-2xl font-bold text-emerald-900">
                                    ₹{bookingAmount}
                                </Text>
                            </View>
                            <TouchableOpacity 
                                onPress={handlePayment}
                                className="bg-emerald-700 px-5 py-3 rounded-lg shadow-sm flex-row items-center"
                            >
                                <MaterialIcons name="payment" size={18} color="white" />
                                <Text className="text-white font-semibold ml-2">Confirm Payment</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
};

export default BookingPage;