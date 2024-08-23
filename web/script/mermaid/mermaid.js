
function updateGraphics() {
    const editorContent = document.getElementById("mermaid-editor").value;
    // 配置 Mermaid
    mermaid.initialize({
        startOnLoad: false,
        theme: 'default'  // 你可以选择其他主题
    });
    // 渲染图形
    mermaid.render('mermaid-rendered', editorContent, (svgCode, bindFunctions) => {
        const graphicsContainer = document.getElementById("mermaid-graphics");
        graphicsContainer.innerHTML = svgCode;

        const svgElement = graphicsContainer.querySelector("svg");

        let isDragging = false;
        let prevX = 0, prevY = 0;
        let scale = 1;

        svgElement.addEventListener("mousedown", function(event) {
            isDragging = true;
            prevX = event.clientX;
            prevY = event.clientY;
        });

        window.addEventListener("mousemove", function(event) {
            if (isDragging) {
                let newX = event.clientX;
                let newY = event.clientY;

                const dx = newX - prevX;
                const dy = newY - prevY;

                let currentTransform = svgElement.style.transform;
                const regex = /translate\(([\d\.\-\+]+)px, ([\d\.\-\+]+)px\)/;
                const match = currentTransform.match(regex);

                let x = parseFloat(match ? match[1] : 0);
                let y = parseFloat(match ? match[2] : 0);

                x += dx;
                y += dy;

                svgElement.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;

                prevX = newX;
                prevY = newY;
            }
        });

        window.addEventListener("mouseup", function() {
            isDragging = false;
        });

        svgElement.addEventListener("wheel", function(event) {
            event.preventDefault();
            scale += event.deltaY * -0.001;

            // Restrict scale
            scale = Math.min(Math.max(.125, scale), 4);

            // 获取当前的平移参数
            let currentTransform = svgElement.style.transform;
            const regex = /translate\(([\d\.\-\+]+)px, ([\d\.\-\+]+)px\)/;
            const match = currentTransform.match(regex);

            let x = parseFloat(match ? match[1] : 0);
            let y = parseFloat(match ? match[2] : 0);

            // Apply scale transform
            svgElement.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
        });

    });
}
