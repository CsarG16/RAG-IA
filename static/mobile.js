// 📱 AulaIA — ajustes para celular: marca la página e inserta el botón hamburguesa del chat.
(function () {
  function init() {
    var body = document.body;
    var isChat = !!document.getElementById('chatContainer');
    body.classList.add(isChat ? 'aulaia-chat' : 'aulaia-landing');
    if (!isChat) return;

    // Botón ☰ para abrir el sidebar (visible solo < 768px vía mobile.css)
    var navbar = document.getElementById('topNavbar');
    if (navbar && !document.getElementById('mobileMenuBtn')) {
      var btn = document.createElement('button');
      btn.id = 'mobileMenuBtn';
      btn.type = 'button';
      btn.title = 'Abrir menú';
      btn.setAttribute('aria-label', 'Abrir menú');
      btn.innerHTML = '<span class="material-symbols-outlined text-2xl">menu</span>';
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        if (typeof window.toggleSidebar === 'function') window.toggleSidebar();
      });
      var slot = navbar.firstElementChild || navbar;
      slot.insertBefore(btn, slot.firstChild);
    }

    initPullToRefresh();
    initShortPlaceholder();

    // Si se agranda la ventana (rotar tablet), quitar el overlay que quedó abierto
    window.addEventListener('resize', function () {
      var overlay = document.getElementById('sidebarOverlay');
      if (overlay && window.innerWidth >= 1024) overlay.classList.add('hidden');
    });
  }

  // ------------------------------------------------------------------
  // Arrastrar hacia abajo para recargar (pull-to-refresh).
  // El chat es una "app" de altura fija: la página no hace scroll, solo
  // #chatContainer, así que el gesto nativo del navegador no se activa.
  // Recargar conserva la conversación porque la URL lleva ?c=<id>.
  // ------------------------------------------------------------------
  function initPullToRefresh() {
    var scroller = document.getElementById('chatContainer');
    if (!scroller || !('ontouchstart' in window)) return;

    var THRESHOLD = 75;   // px (ya con resistencia) para disparar la recarga
    var MAX_PULL = 110;
    var startY = null, startX = 0, pull = 0, armed = false;

    var ind = document.createElement('div');
    ind.id = 'pullRefreshIndicator';
    ind.setAttribute('aria-hidden', 'true');
    ind.innerHTML = '<span class="material-symbols-outlined">refresh</span>';
    document.body.appendChild(ind);

    function busy() {
      try { if (typeof isStreaming !== 'undefined' && isStreaming) return true; } catch (e) {}
      var quiz = document.getElementById('quizModal');
      return !!(quiz && !quiz.classList.contains('hidden'));
    }

    function paint(px, animate) {
      ind.style.transition = animate ? 'transform 220ms ease, opacity 220ms ease' : 'none';
      var progress = Math.min(px / THRESHOLD, 1);
      ind.style.opacity = px > 4 ? String(0.35 + progress * 0.65) : '0';
      ind.style.transform = 'translate(-50%, ' + (px - 48) + 'px) rotate(' + (progress * 270) + 'deg)';
      ind.classList.toggle('ready', px >= THRESHOLD);
    }

    scroller.addEventListener('touchstart', function (e) {
      if (e.touches.length !== 1 || scroller.scrollTop > 0 || busy()) { startY = null; return; }
      startY = e.touches[0].clientY;
      startX = e.touches[0].clientX;
      pull = 0;
      armed = false;
    }, { passive: true });

    scroller.addEventListener('touchmove', function (e) {
      if (startY === null) return;
      var dy = e.touches[0].clientY - startY;
      var dx = Math.abs(e.touches[0].clientX - startX);
      if (scroller.scrollTop > 0 || dy <= 0 || dx > dy) { pull = 0; paint(0, false); return; }
      pull = Math.min(dy * 0.5, MAX_PULL);   // resistencia
      armed = pull >= THRESHOLD;
      paint(pull, false);
    }, { passive: true });

    function end() {
      if (startY === null) return;
      startY = null;
      if (armed) {
        ind.classList.add('spinning');
        paint(THRESHOLD, true);
        setTimeout(function () { window.location.reload(); }, 250);
      } else {
        paint(0, true);
      }
    }
    scroller.addEventListener('touchend', end, { passive: true });
    scroller.addEventListener('touchcancel', end, { passive: true });
  }

  // En celular el texto de ayuda del cuadro de preguntas es corto (dos palabras)
  function initShortPlaceholder() {
    var input = document.getElementById('userInput');
    if (!input) return;
    var longText = input.getAttribute('placeholder') || '';
    var mq = window.matchMedia('(max-width: 767px)');
    function apply() {
      input.setAttribute('placeholder', mq.matches ? 'Pregunta algo' : longText);
    }
    apply();
    if (mq.addEventListener) mq.addEventListener('change', apply);
    else if (mq.addListener) mq.addListener(apply);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
