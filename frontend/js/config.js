/**
 * LuminaPV - Configuration & Constants
 * Defines API routing, inspection parameters, defect classification tokens, and DOM utility selectors.
 */

var API_ENDPOINT = "/api/v1/analyze/bulk";
var CLUSTER_ENDPOINT = "/api/v1/cluster/status";
var CLUSTER_POLL_INTERVAL_MS = 4000;
var AUTH_LOGIN_ENDPOINT = "/api/auth/login";
var AUTH_REGISTER_ENDPOINT = "/api/auth/register";
var AUTH_LOGOUT_ENDPOINT = "/api/auth/logout";
var AUTH_ME_ENDPOINT = "/api/auth/me";
var HISTORY_ENDPOINT = "/api/v1/user/history";
var HISTORY_SYNC_ENDPOINT = "/api/v1/user/history/sync";
var LOCAL_MODULES_KEY = "luminapv.local.modules";
var SAMPLE_PAYLOAD = "fixtures/luminapv-batch-sample.json";
var TIMEOUT_MS = 60000;
var SLOW_MS = 15000;

var DEFECT_LABEL = {
  "micro-crack": "MICROTRINCA",
  "dead": "CÉLULA MORTA",
  "hot-spot": "PONTO QUENTE",
  "delamination": "DELAMINAÇÃO",
  "grid-fault": "FALHA DE GRADE",
  "defeituosa": "DEFEITUOSA",
  "none": "—"
};

var DEFECT_VAR = {
  "micro-crack": "--d1",
  "dead": "--d2",
  "hot-spot": "--d3",
  "delamination": "--d4",
  "grid-fault": "--d5",
  "defeituosa": "--d1"
};

var DEFECT_KEYS = ["micro-crack", "dead", "hot-spot", "delamination", "grid-fault", "defeituosa"];

var $ = function(s, r) { return (r || document).querySelector(s); };
var $$ = function(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
