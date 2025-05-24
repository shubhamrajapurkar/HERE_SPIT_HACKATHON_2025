import Map from "./components/Map";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Items from './components/Items';
import { HereProvider } from "./context/HereContext";
import RoutingPage from './components/RoutingPage';
import RoutePlanner from './components/RoutePlanner';
import UserReviewsPage from './components/UserReviewsPage';
import PositiveReviewPage from './components/PositiveReviewPage';
import NegativeReviewPage from "./components/NegativeReviewPage";



function App() {
	return (
		<HereProvider>
			<BrowserRouter>
				<div className="flex h-screen w-full">
					<Items />
					<div className="flex-1 pl-16 w-full">
						<Routes>
							<Route path="/" element={<Map />} />
							<Route path="/route" element={<RoutingPage />} />
							<Route path="/route-planner" element={<RoutePlanner />} />
							<Route path="/user-reviews" element={<UserReviewsPage />} />
						</Routes>
					</div>
				</div>
			</BrowserRouter>
		</HereProvider>
	);
}

export default App;