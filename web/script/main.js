const user_id = localStorage.getItem("user_id");
let reconnectInterval = 1000; // 1 second
if(!localStorage.getItem('access_token'))
   login();

function connect() {
    if (user_id) {
        const socket = new WebSocket(`ws://localhost:8000/ws/${user_id}`);
        let heartbeatInterval;

        socket.onopen = () => {
            console.log('Connected to server');
            reconnectInterval = 1000; // Reset the interval on successful connection

            // Start sending heartbeat messages every 10 seconds
            heartbeatInterval = setInterval(() => {
                if (socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ type: 'heartbeat' }));
                }
            }, 10000);
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type !== 'heartbeat') {
                const url = `http://localhost:${data.port}${data.mount_path}`;
                addTabWithUrl(url, data.name);
            }
        };

        socket.onclose = (event) => {
            console.log('Connection closed', event);
            clearInterval(heartbeatInterval); // Stop heartbeat messages
            // Attempt to reconnect after a delay
            setTimeout(connect, reconnectInterval);
            // Increment the interval for each failed attempt
            reconnectInterval = Math.min(reconnectInterval * 2, 5000); // Max 5 seconds
        };

        socket.onerror = (error) => {
            console.log('WebSocket error', error);
        };
    }
}
connect();