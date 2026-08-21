// The number of pixels the user must scroll by before the logo is hidden.
const scrollTopPixels = 234;

// Menu margin when the search bar is fixed. Roughly the logo's height, so the
// top menu items are not hidden behind it.
const menuTopMargin = '330px';

// Hide the navigation bar logo when scrolling down on desktop: it is tall
// enough to crowd the rest of the bar.
function registerOnScrollEvent(mediaQuery) {
  // The navigation bar that contains the logo.
  const $navbar = $('.wy-side-scroll');
  const $menu = $('.wy-menu-vertical');
  const $search = $('.wy-side-nav-search');

  // The anchor containing the logo; hide that rather than the logo itself, or a
  // small clickable area stays visible.
  const $logo = $('.wy-side-nav-search > a');

  if (mediaQuery.matches) {
    // We're on desktop; register the scroll event.
    $navbar.scroll(function() {
      if ($(this).scrollTop() >= scrollTopPixels) {
        $logo.hide();
        $search.addClass('fixed');
        $menu.css('margin-top', menuTopMargin);
      } else {
        $logo.show();
        $search.removeClass('fixed');
        $menu.css('margin-top', 0);
      }
    });
  } else {
    // We're on mobile; unregister the scroll event so the logo isn't hidden
    // when scrolling.
    $logo.show();
    $navbar.unbind('scroll');
  }
}

$(document).ready(() => {
  const mediaQuery = window.matchMedia('only screen and (min-width: 768px)');
  registerOnScrollEvent(mediaQuery);
  mediaQuery.addListener(registerOnScrollEvent);
});
