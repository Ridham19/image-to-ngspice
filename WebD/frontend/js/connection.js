/**
 * Connection checking utility for WebSpice Studio.
 * Periodically checks the backend status and updates the UI badge.
 */
async function checkBackendConnection() {
    const badge = document.getElementById('backendConnection');
    if (!badge) return;
    const text = badge.querySelector('.connection-text');
    if (!text) return;
    
    // Set status to checking
    if (!badge.classList.contains('status-checking')) {
        badge.className = 'connection-badge status-checking';
        text.textContent = 'Backend: Checking';
    }
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000); // 3-second timeout
    
    try {
        const response = await fetch('http://127.0.0.1:8000/', {
            method: 'GET',
            signal: controller.signal,
            mode: 'cors'
        });
        clearTimeout(timeoutId);
        
        if (response.ok) {
            badge.className = 'connection-badge status-online';
            text.textContent = 'Backend: Connected';
        } else {
            badge.className = 'connection-badge status-offline';
            text.textContent = 'Backend: Error';
        }
    } catch (error) {
        clearTimeout(timeoutId);
        badge.className = 'connection-badge status-offline';
        text.textContent = 'Backend: Offline';
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    checkBackendConnection();
    
    // Let user click to force check
    const badge = document.getElementById('backendConnection');
    if (badge) {
        badge.addEventListener('click', checkBackendConnection);
    }
});
