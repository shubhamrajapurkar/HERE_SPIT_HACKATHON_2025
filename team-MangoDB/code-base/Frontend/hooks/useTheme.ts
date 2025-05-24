import { useColorScheme } from 'react-native';

export function useTheme() {
  const colorScheme = useColorScheme();
  
  return {
    isDark: colorScheme === 'dark',
    colors: {
      background: colorScheme === 'dark' ? '#121212' : '#FFFFFF',
      surface: colorScheme === 'dark' ? '#1E1E1E' : '#F3F4F6',
      text: colorScheme === 'dark' ? '#FFFFFF' : '#000000',
      primary: colorScheme === 'dark' ? '#34D399' : '#065F46',
      secondary: colorScheme === 'dark' ? '#374151' : '#E5E7EB',
    }
  };
}