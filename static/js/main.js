( () => {
    "use strict";
    var e = {
        r: e => {
            "undefined" != typeof Symbol && Symbol.toStringTag && Object.defineProperty(e, Symbol.toStringTag, {
                value: "Module"
            }),
            Object.defineProperty(e, "__esModule", {
                value: !0
            })
        }
    }
      , t = {};
    e.r(t);
    const c = Object.freeze({
        latest: "3.74.1_3.2.13-170510df",
        public: "3.74.1_3.2.12-59b2719c",
        legacy: "3.74.1_3.2.12-59b2719c",
        edit: "2.24.4_2.12.0-7d0a8c15"
    });
    !function loadSDKScript() {
        var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : "latest.js"
          , t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : c.latest
          , n = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : "View"
          , r = Array.from(document.scripts)
          , i = "view-sdk/".concat(e)
          , o = r.find((function(e) {
            return e.src && e.src.endsWith(i)
        }
        )).src
          , d = o.substring(0, o.indexOf(e))
          , a = "".concat("".concat(d + t, "/").concat(n), "SDKInterface.js")
          , s = document.createElement("script");
        s.async = !1,
        s.setAttribute("src", a),
        document.head.appendChild(s)
    }("main.js", c.legacy),
    window.adobe_dc_view_sdk = t
}
)();
//# sourceMappingURL=3.74.1_3.2.13-170510df/private/main.js.map
