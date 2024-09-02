
// 执行检查
if (!!window.chrome) {
    checkExtension().then((response) => {
        if (response && response.user_id && response.access_token) {
            console.log('Extension is installed:', response);
            localStorage.setItem("user_id", response.user_id);
            localStorage.setItem("access_token", response.access_token);
        } else {
            console.log('Extension not installed or not responding, or received invalid data');
            if (!localStorage.getItem('access_token')) {
                login();  // 如果没有登录信息，调用 login 函数
            }
        }
    });
} else {
    console.warn("This feature is designed to work with Google Chrome only.");
    if (!localStorage.getItem('access_token')) {
        login();
    }
}
function login() {
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded', // 修改为表单数据类型
        },
        body: new URLSearchParams({
            username: 'testuser',
            password: 'testpassword',
        }),
    })
    .then(
        response => response.json()
    ).then(data => {
        const token = data.access_token;
        const user_id = data.user_id;
        localStorage.setItem('access_token', token);
        localStorage.setItem('user_id', user_id);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}