function showContent(frame_id,menu_class_name,page, element) {
    // Update the iframe's src attribute to load the corresponding page
    document.getElementById(frame_id).src = page;
    // Update the active state of the menu items
    var menuItems = document.getElementsByClassName(menu_class_name);
    for (var i = 0; i < menuItems.length; i++) {
        menuItems[i].classList.remove('active');
    }
    element.classList.add('active');
}

function loadAxios() {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js';
        script.onload = function () {
            // 在 axios 加载完毕后，设置拦截器
            const axiosInstance = axios.create();

            axiosInstance.interceptors.request.use(
                function (config) {
                    const token = localStorage.getItem('access_token');
                    if (token) {
                        config.headers.Authorization = 'Bearer ' + token;
                    }
                    return config;
                },
                function (error) {
                    return Promise.reject(error);
                }
            );

            resolve(axiosInstance); // 返回设置了拦截器的 axios 实例
        };
        script.onerror = function () {
            reject(new Error('Failed to load axios'));
        };
        document.head.appendChild(script);
    });
}

function makeRequest({ method, url, params }, { onSuccess, onError, clientParams, responseParamsExtractor }) {
    loadAxios().then(function (axiosInstance) {
        let request;
        if (method === 'get') {
            request = axiosInstance.get(url, { params });
        } else if (method === 'post') {
            request = axiosInstance.post(url, params, {
                headers: { 'Content-Type': 'application/json' }
            });
        }

        request.then(function (response) {
            if (response.status === 200) {
                const responseParams = responseParamsExtractor(response);
                onSuccess(clientParams, responseParams);
            } else {
                alert('HTTP error! status: ' + response.status);
                throw new Error('HTTP error! status: ' + response.status);
            }
        }).catch(function (e) {
            if (onError) {
                onError(e);
            } else {
                alert('Error: ' + e.message);
                console.error('Error:', e);
            }
        });
    }).catch(function (error) {
        alert('Failed to load axios: ' + error.message);
        console.error('Error:', error);
    });
}
