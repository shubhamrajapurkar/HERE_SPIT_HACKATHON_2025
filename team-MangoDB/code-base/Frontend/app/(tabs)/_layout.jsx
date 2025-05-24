import React from "react";
import { Tabs } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { StatusBar } from "expo-status-bar";

export default function Layout() {
  return (
    <>
      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarShowLabel: true,
          tabBarStyle: {
            height: 70, // Increased height for more breathing room
            backgroundColor: "white",
            borderTopWidth: 1,
            borderTopColor: "#e0e0e0",
            paddingVertical: 10, // Added vertical padding for spacing
            paddingHorizontal: 12, // Added horizontal padding for symmetry
            shadowColor: "#000", // Subtle shadow for a floating effect
            shadowOpacity: 0.1,
            shadowOffset: { width: 0, height: -2 },
            shadowRadius: 10,
            elevation: 5,
          },
          tabBarLabelStyle: {
            fontSize: 12, // Slightly increased font size
            fontWeight: "600", // Bold for better readability
            marginBottom: 8, // More space between icon and label
          },
          tabBarInactiveTintColor: "#8e8e93",
          tabBarActiveTintColor: "#065f46",
          tabBarItemStyle: {
            marginHorizontal: 10, // Added horizontal spacing between tabs
          },
          tabBarIconStyle: {
            marginTop: 4, // Additional spacing above the icons
          },
        }}
      >
        {/* Home Tab */}
        <Tabs.Screen
          name="home"
          options={{
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="home" color={color} size={24} />
            ), // Slightly larger icon
            tabBarLabel: "Home",
          }}
        />

        {/* Travel Tab */}
        <Tabs.Screen
          name="travel"
          options={{
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="airplane" color={color} size={24} />
            ),
            tabBarLabel: "Travel",
            headerShown: false,
            headerTitle: "Where To?",
            headerTitleAlign: "center",
            headerStyle: {
              backgroundColor: "#fff",
              height: 80,
            },
            headerTitleStyle: {
              fontWeight: "bold",
              fontSize: 18,
            },
          }}
        />

        {/* Tickets Tab */}
        <Tabs.Screen
          name="tickets"
          options={{
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="ticket" color={color} size={24} />
            ),
            tabBarLabel: "Tickets",
            headerShown: true, // Show header
            headerTitle: "Your Tickets", // Set header title
            headerTitleAlign: "center",
            headerStyle: {
              backgroundColor: "#fff",
              height: 80,
            },
            headerTitleStyle: {
              fontWeight: "bold",
              fontSize: 18,
            },
          }}
        />

        {/* Account Tab */}
        <Tabs.Screen
          name="account"
          options={{
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="person" color={color} size={24} />
            ),
            tabBarLabel: "Safety",
          }}
        />
      </Tabs>
      <StatusBar backgroundColor="#fcfcfc" style="dark" />
    </>
  );
}
