import React from 'react';
  import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
  import { createDrawerNavigator } from '@react-navigation/drawer';
  import { NavigationContainer } from '@react-navigation/native';
  import { Ionicons } from '@expo/vector-icons';

  import HomeScreen from './screens/HomeScreen';
  import EmergencyScreen from './screens/EmergencyScreen';
  import ProfileScreen from './screens/ProfileScreen';
  import NearbyPlacesScreen from './screens/NearbyPlacesScreen';
  import EnableJourneyScreen from './screens/EnableJourneyScreen';
import MapScreen from './screens/MapScreen';
import TrafficScreen from './screens/TrafficScreen';
import RouteDetailsScreen from './screens/RouteDetailsScreen ';

  const Tab = createBottomTabNavigator();
  const Drawer = createDrawerNavigator();

  function BottomTabs() {
    return (
      <Tab.Navigator
        screenOptions={({ route }) => ({
          headerShown: false,
          tabBarIcon: ({ color, size }) => {
            let iconName;
            if (route.name === 'Home') iconName = 'home';
            else if (route.name === 'Emergency') iconName = 'alert-circle';
            else if (route.name === 'Profile') iconName = 'person';

            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: 'tomato',
          tabBarInactiveTintColor: 'gray',
        })}
      >
        <Tab.Screen name="Home" component={HomeScreen} />
        {/* <Tab.Screen name="Emergency" component={EmergencyScreen} /> */}
        <Tab.Screen name="Profile" component={ProfileScreen} />
      </Tab.Navigator>
    );
  }

  export default function AppNavigator() {
    return (
      <NavigationContainer>
        <Drawer.Navigator initialRouteName="Main">
          <Drawer.Screen name="Main" component={BottomTabs} />
          <Drawer.Screen name="Map" component={MapScreen} />
          <Drawer.Screen name="Emergency" component={EmergencyScreen} />
           <Drawer.Screen name="Traffic" component={TrafficScreen} />
          <Drawer.Screen name="Route-details" component={RouteDetailsScreen} />
        </Drawer.Navigator>
      </NavigationContainer>
    );
  }

