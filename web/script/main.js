function showContent(page, element) {
    // Update the iframe's src attribute to load the corresponding page
    document.getElementById('content-frame').src = page;

    // Update the active state of the menu items
    var menuItems = document.getElementsByClassName('menu-item');
    for (var i = 0; i < menuItems.length; i++) {
        menuItems[i].classList.remove('active');
    }
    element.classList.add('active');
}



