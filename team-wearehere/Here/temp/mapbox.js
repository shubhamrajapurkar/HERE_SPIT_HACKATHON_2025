mapboxgl.accessToken ="pk.eyJ1Ijoic3ViaGFtcHJlZXQiLCJhIjoiY2toY2IwejF1MDdodzJxbWRuZHAweDV6aiJ9.Ys8MP5kVTk5P9V2TDvnuDg";

navigator.geolocation.getCurrentPosition(successLocation, errorLocation, {
    enableHighAccuracy: true
  })
  
  function successLocation(position) {
    setupMap([position.coords.longitude, position.coords.latitude])
  }
  
  function errorLocation() {
    setupMap([-2.24, 53.48])
  }
  
  function setupMap(center) {
    const map = new mapboxgl.Map({
      container: "map",
      style: "mapbox://styles/mapbox/streets-v11",
      center: center,
      zoom: 15
    })
  
    const nav = new mapboxgl.NavigationControl()
    map.addControl(nav)
  
    var directions = new MapboxDirections({
      accessToken: mapboxgl.accessToken
    })
  
    map.addControl(directions, "top-left")

    // Prefill directions if available
    if (window.prefillDirections) {
      const src = window.prefillDirections.src;
      const dest = window.prefillDirections.dest;
      // Try to load cached route
      const cacheKey = `routeCache:${src}->${dest}`;
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        try {
          const { origin, destination } = JSON.parse(cached);
          directions.setOrigin(origin);
          directions.setDestination(destination);
        } catch (e) {
          directions.setOrigin(src);
          directions.setDestination(dest);
        }
      } else {
        directions.setOrigin(src);
        directions.setDestination(dest);
      }
      // Cache the route after it's loaded
      directions.on('route', function() {
        localStorage.setItem(cacheKey, JSON.stringify({
          origin: src,
          destination: dest
        }));
      });
    }

    // Expose for explore.js
    window.mapboxMap = map;
    window.mapboxDirections = directions;
  }
